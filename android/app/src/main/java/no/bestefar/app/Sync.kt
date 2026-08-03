package no.bestefar.app

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Opportunistisk opplasting av køen i `filesDir/dev_uploads` til
 * `/v1/failed-analyses` (backend_spec.md §6).
 *
 * Køen er filbasert med vilje: den overlever appdrap, omstart og flymodus, og
 * krever ingen database. Hvert element er et par
 * `{seriesId}_{tag}.jpg` + `{seriesId}_{tag}.json`.
 *
 * Sending skjer ved appstart og på «Send nå» — ikke som bakgrunnsjobb. Det er
 * nok for offline-først (§0) og sparer oss for WorkManager; trenger vi sending
 * mens appen er lukket, er et periodisk WorkManager-arbeid neste steg.
 */
object Sync {

    private const val TAG = "BestefarSync"
    /** Format på sidecar-JSON. 1 = før backend-kobling (kun detected + tag). */
    private const val SIDECAR_VERSION = 2
    /** Sendes når en gammel kø-fil ikke har konfidens. Backenden lagrer den rått. */
    private const val UNKNOWN_CONFIDENCE = -1.0

    private val running = AtomicBoolean(false)

    fun dir(ctx: Context): File = File(ctx.filesDir, "dev_uploads")

    /** Antall elementer som venter på å bli sendt. */
    fun pending(ctx: Context): Int =
        dir(ctx).listFiles { f -> f.name.endsWith(".json") }?.size ?: 0

    /**
     * Legg en innsending i køen. Kalles på UI-tråden — filene er små, og
     * bildet er allerede skrevet til disk av CaptureActivity.
     */
    fun queue(ctx: Context, seriesId: String, tag: String, statusCode: Int,
              confidence: Double, detected: List<Double>, ocr: List<Double>?,
              imagePath: String?) {
        try {
            val d = dir(ctx).apply { mkdirs() }
            imagePath?.let { p ->
                val src = File(p)
                if (src.exists()) src.copyTo(File(d, "${seriesId}_$tag.jpg"), overwrite = true)
            }
            val meta = JSONObject().apply {
                put("v", SIDECAR_VERSION)
                put("series_id", seriesId)
                put("tag", tag)
                put("status_code", statusCode)
                put("confidence", confidence)
                put("core_version", BuildConfig.VERSION_NAME)
                put("detected", JSONArray(detected))
                ocr?.let { put("ocr", JSONArray(it)) }
            }
            File(d, "${seriesId}_$tag.json").writeText(meta.toString())
        } catch (e: Exception) {
            // Best effort: en tapt donasjon skal aldri velte lagringsflyten.
            Log.w(TAG, "Kunne ikke køe $seriesId/$tag", e)
        }
    }

    /** `skipped` = vi prøvde ikke (offline, eller kun-wifi og vi er på mobildata). */
    class Outcome(val sent: Int, val failed: Int, val dropped: Int, val skipped: Boolean) {
        val tried: Int get() = sent + failed + dropped
    }

    /**
     * Tøm køen. [force] hopper over kun-wifi-innstillingen (brukeren trykket
     * «Send nå» og vet hva de gjør) — men ikke over «er vi på nett i det hele
     * tatt», for da hadde vi bare ventet ut en timeout.
     */
    fun flush(ctx: Context, force: Boolean = false, onDone: ((Outcome) -> Unit)? = null) {
        val app = ctx.applicationContext
        val store = Store.get(app)
        val files = dir(app).listFiles { f -> f.name.endsWith(".json") }
        if (files == null || files.isEmpty()) {
            onDone?.let { Api.ui { it(Outcome(0, 0, 0, false)) } }
            return
        }
        if (!Api.online(app, wifiOnly = store.uploadWifiOnly && !force)) {
            onDone?.let { Api.ui { it(Outcome(0, 0, 0, true)) } }
            return
        }
        if (!running.compareAndSet(false, true)) {
            // En tømming holder allerede på. Meld tilbake likevel, ellers blir
            // «Send nå»-knappen stående deaktivert i påvente av et svar som
            // aldri kommer.
            onDone?.let { Api.ui { it(Outcome(0, 0, 0, false)) } }
            return
        }

        Api.io {
            var sent = 0; var failed = 0; var dropped = 0
            try {
                for (meta in files.sortedBy { it.lastModified() }) {
                    when (send(app, meta)) {
                        Verdict.SENT -> sent++
                        Verdict.DROP -> dropped++
                        Verdict.KEEP -> failed++
                    }
                }
                dropOrphanImages(app)
                store.lastSyncTs = System.currentTimeMillis()
            } finally {
                running.set(false)
            }
            onDone?.let { cb -> Api.ui { cb(Outcome(sent, failed, dropped, false)) } }
        }
    }

    private enum class Verdict { SENT, KEEP, DROP }

    private fun send(ctx: Context, meta: File): Verdict {
        val o = try {
            JSONObject(meta.readText())
        } catch (e: Exception) {
            Log.w(TAG, "Ubrukelig kø-fil ${meta.name}, kastes", e)
            remove(meta); return Verdict.DROP
        }
        val image = File(meta.parentFile, meta.nameWithoutExtension + ".jpg")
        if (!image.exists()) {
            // Endepunktet krever bildet; uten det er metadataene verdiløse.
            remove(meta); return Verdict.DROP
        }

        val fields = mapOf(
            "status_code" to o.optInt("status_code", BestefarCore.OK).toString(),
            "confidence" to o.optDouble("confidence", UNKNOWN_CONFIDENCE).toString(),
            "core_version" to o.optString("core_version", "ukjent"),
            "tag" to o.optString("tag", "rejected"),
            "series_id" to o.optString("series_id", meta.nameWithoutExtension),
            "detected_scores" to (o.optJSONArray("detected") ?: JSONArray()).toString(),
            "ocr_scores" to (o.optJSONArray("ocr") ?: JSONArray()).toString(),
        )
        val resp = Api.postMultipart(ctx, "/v1/failed-analyses", fields,
            fileField = "image", file = image, mime = "image/jpeg")
        return when {
            resp.ok -> { remove(meta); Verdict.SENT }
            // 4xx som ikke er 408/429: for stort bilde, ugyldig felt, feil tag.
            // Dette blir aldri bedre av å prøve igjen — ikke la køen gro.
            !resp.retryable -> {
                Log.w(TAG, "Kø-element ${meta.name} avvist (${resp.code}): ${resp.body.take(200)}")
                remove(meta); Verdict.DROP
            }
            else -> Verdict.KEEP
        }
    }

    private fun remove(meta: File) {
        File(meta.parentFile, meta.nameWithoutExtension + ".jpg").delete()
        meta.delete()
    }

    /** JPEG uten sidecar kan ikke sendes (endepunktet krever metadataene). */
    private fun dropOrphanImages(ctx: Context) {
        dir(ctx).listFiles { f -> f.name.endsWith(".jpg") }?.forEach { img ->
            if (!File(img.parentFile, img.nameWithoutExtension + ".json").exists()) img.delete()
        }
    }
}
