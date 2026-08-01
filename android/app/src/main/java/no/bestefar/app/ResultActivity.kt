package no.bestefar.app

import android.content.Intent
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton
import java.io.File
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Resultatkort (musingsUI): skive med treffene markert, poenglista til høyre og
 * «Ikke lagre» / «OK» under. Runde 4: OCR-finpussing av poeng, innskytingssjekk
 * på sesongens første serie, varsel om identiske serier. Runde 10: blyanten er
 * fjernet — OCR har overtatt korreksjonsrollen, og feiler analysen ber vi heller
 * om et nytt bilde.
 */
class ResultActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_STATUS = "status"
        const val EXTRA_SUM_DEC = "sum_dec"
        const val EXTRA_SUM_INT = "sum_int"
        const val EXTRA_N_HITS = "n_hits"
        const val EXTRA_CONFIDENCE = "confidence"
        const val EXTRA_DECIMALS = "decimals"
        const val EXTRA_INTEGERS = "integers"
        const val EXTRA_RREL = "r_rel"
        const val EXTRA_THETA = "theta"
        const val EXTRA_IMAGE_PATH = "image_path"
        const val EXTRA_GALLERY_URI = "gallery_uri"
        /** «8. mars 2026    08:38» — god luft mellom dato og tid (musingsUI r7). */
        val DATE_TIME_FMT: java.time.format.DateTimeFormatter =
            java.time.format.DateTimeFormatter.ofPattern("d. MMMM yyyy    HH:mm",
                java.util.Locale("no"))
        /**
         * Maks avstand mellom to treffpunkter for at de skal regnes som «samme
         * treff» ved duplikatsjekk (musingsUI runde 10). rRel er oppgitt i
         * RINGSTEG, og ett ringsteg = ett poeng, så 0,1 her = 0,1 poeng.
         */
        const val SAME_HIT_POINTS = 0.1
    }

    private lateinit var store: Store
    private lateinit var content: LinearLayout
    private var record: SeriesRecord? = null
    private var saved = false
    private var imagePath: String? = null
    private var galleryUri: Uri? = null
    /** OCR-poeng i SKJERMREKKEFØLGE (musingsUI runde 10). Null = ingen OCR. */
    private var ocrScores: List<Double>? = null
    /** Falsk når OCR og de detekterte treffene er uenige (> 0.2). */
    private var ocrAgrees = true

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = Store.get(this)
        content = Ui.col(this)
        val scroller = Ui.scroll(this, content)
        Ui.applyInsets(scroller)
        setContentView(scroller)
        imagePath = intent.getStringExtra(EXTRA_IMAGE_PATH)
        galleryUri = intent.getStringExtra(EXTRA_GALLERY_URI)?.let { Uri.parse(it) }

        val status = intent.getIntExtra(EXTRA_STATUS, BestefarCore.ERROR_INTERNAL)
        if (status != BestefarCore.OK) {
            renderRejected(status)
            return
        }

        savedInstanceState?.getString("recordId")?.let { id ->
            record = store.allSeries().firstOrNull { it.id == id }
            if (record != null) { saved = true; render(); return }
        }

        val decimals = intent.getDoubleArrayExtra(EXTRA_DECIMALS) ?: doubleArrayOf()
        val integers = intent.getIntArrayExtra(EXTRA_INTEGERS)
            ?: decimals.map { it.toInt() }.toIntArray()
        val rrel = intent.getDoubleArrayExtra(EXTRA_RREL) ?: DoubleArray(decimals.size)
        val theta = intent.getDoubleArrayExtra(EXTRA_THETA) ?: DoubleArray(decimals.size)
        val shots = decimals.indices.map {
            Shot(decimals[it], integers[it], rrel[it], theta[it])
        }

        // «Vil du lagre skjermbildet i bildearkivet?» etter FØRSTE (vellykkede)
        // scan (musingsUI runde 10) — før stillingsvalget, så spørsmålet ikke
        // kommer midt i lagringsflyten.
        askGallerySave {
            resolvePosition { pos, mod ->
                val w = store.selectedWeapon()
                record = SeriesRecord(
                    id = Store.newId(),
                    ts = System.currentTimeMillis(),
                    weaponId = w?.id,
                    ammoName = w?.ammoName ?: "",
                    distanceM = store.distanceM,
                    position = pos,
                    modifier = mod,
                    shots = shots,
                )
                // Innskytingssjekk før OCR/visning (kan forkaste serien)
                sightInCheck { runOcrThenRender() }
            }
        }
    }

    // ---------- Lagring av scan i bildearkivet (musingsUI runde 10) ----------

    /**
     * Spør én gang — etter første scan — om skjermbildet skal ligge i brukerens
     * bildearkiv, og forklar hvor valget kan endres. Svarer man «Nei» slettes
     * også bildet fra denne scanen, siden CaptureActivity lagrer først og spør
     * etterpå (bildet er det mest verdifulle vi har hvis analysen bommer).
     */
    private fun askGallerySave(onDone: () -> Unit) {
        if (store.saveScansAsked) { onDone(); return }
        store.saveScansAsked = true
        AlertDialog.Builder(this)
            .setMessage(R.string.gallery_save_ask)
            .setPositiveButton(R.string.yes) { _, _ ->
                store.saveScansToGallery = true; galleryInfo(onDone)
            }
            .setNegativeButton(R.string.no) { _, _ ->
                store.saveScansToGallery = false
                deleteGalleryImage()
                galleryInfo(onDone)
            }
            .setCancelable(false)
            .show()
    }

    private fun galleryInfo(onDone: () -> Unit) {
        AlertDialog.Builder(this)
            .setMessage(R.string.gallery_save_info)
            .setPositiveButton(R.string.ok) { _, _ -> onDone() }
            .setOnCancelListener { onDone() }
            .show()
    }

    private fun deleteGalleryImage() {
        val uri = galleryUri ?: return
        try { contentResolver.delete(uri, null, null) } catch (_: Exception) { }
        galleryUri = null
    }

    /** Stilling velges alltid etter scan nå (musingsUI runde 4). */
    private fun resolvePosition(onDone: (Position, PosModifier) -> Unit) {
        Dialogs.positionSheet(this, store, onDone)
    }

    // ---------- Innskyting (musingsUI runde 4) ----------

    private fun sightInCheck(proceed: () -> Unit) {
        val r = record ?: return proceed()
        val shots = r.shots
        if (!Stats.looksMiscalibrated(shots)) return proceed()

        val prior = store.seasonSeriesForWeapon(r.weaponId)
        val today = store.seriesToday().filter { it.weaponId == r.weaponId }

        when {
            // Sesongens (og våpenets) aller første serie
            prior.isEmpty() -> AlertDialog.Builder(this)
                .setTitle(R.string.sightin_title)
                .setMessage(R.string.sightin_body)
                .setPositiveButton(R.string.yes) { _, _ ->
                    Toast.makeText(this, R.string.sightin_registered, Toast.LENGTH_SHORT).show()
                    proceed()
                }
                .setNegativeButton(R.string.no) { _, _ ->
                    Toast.makeText(this, R.string.sightin_discarded, Toast.LENGTH_SHORT).show()
                    finish()
                }
                .setCancelable(false)
                .show()

            // Dagens to første serier med omtrent samme bias
            today.size == 1 && Stats.looksMiscalibrated(today[0].shots) &&
                Stats.similarBias(shots, today[0].shots) -> AlertDialog.Builder(this)
                .setTitle(R.string.sightin_title)
                .setMessage(R.string.sightin_body)
                .setPositiveButton(R.string.yes) { _, _ -> proceed() }
                .setNegativeButton(R.string.no) { _, _ -> askSaveLastTwo(today[0]) { proceed() } }
                .setCancelable(false)
                .show()

            else -> proceed()
        }
    }

    private fun askSaveLastTwo(firstToday: SeriesRecord, proceed: () -> Unit) {
        AlertDialog.Builder(this)
            .setTitle(R.string.sightin_savetwo_title)
            .setPositiveButton(R.string.sightin_save_anyway) { _, _ -> proceed() }
            .setNegativeButton(R.string.sightin_dont_save) { _, _ ->
                // Forkast begge: slett dagens første og ikke lagre denne
                store.deleteSeries(listOf(firstToday.id))
                Toast.makeText(this, R.string.sightin_discarded, Toast.LENGTH_SHORT).show()
                finish()
            }
            .setCancelable(false)
            .show()
    }

    // ---------- OCR-finpussing (musingsUI runde 4) ----------

    private fun runOcrThenRender() {
        val r = record ?: return
        val path = imagePath
        val bmp = if (path != null && File(path).exists())
            try { BitmapFactory.decodeFile(path) } catch (_: Exception) { null } else null
        if (bmp == null) { render(); return }

        OcrVerifier.verify(bmp, r.shots.map { it.decimal }) { result ->
            when (result) {
                is OcrVerifier.Result.Match -> {
                    // Sømløs oppdatering til OCR-poengene (sortert mapping)
                    ocrScores = result.scores; ocrAgrees = true
                    applyScores(result.scores)
                    if (store.shareDevImagesActive) queueDevImage(r, "ocr_match")
                    render()
                }
                is OcrVerifier.Result.Mismatch -> {
                    ocrScores = result.scores; ocrAgrees = false
                    render()
                }
                OcrVerifier.Result.Inconclusive -> render()
            }
        }
    }

    /** Erstatt de viste poengene (sortert) med et nytt sett, behold posisjon. */
    private fun applyScores(scores: List<Double>) {
        val r = record ?: return
        val order = r.shots.indices.sortedBy { r.shots[it].decimal }
        val sorted = scores.sorted()
        val newShots = r.shots.toMutableList()
        order.forEachIndexed { rank, idx ->
            val v = sorted[rank]
            newShots[idx] = r.shots[idx].copy(
                decimal = v, integer = floor(v).toInt().coerceIn(0, 10))
        }
        r.shots = newShots
    }

    /** Køer fullskala-bilde + poeng til utvikler (backend-opplasting: TODO). */
    private fun queueDevImage(r: SeriesRecord, tag: String) {
        try {
            val dir = File(filesDir, "dev_uploads").apply { mkdirs() }
            imagePath?.let { p ->
                File(p).copyTo(File(dir, "${r.id}_$tag.jpg"), overwrite = true)
            }
            File(dir, "${r.id}_$tag.json").writeText(
                """{"detected":[${r.shots.joinToString(",") { "%.1f".format(java.util.Locale.US, it.decimal) }}],"tag":"$tag"}""")
        } catch (_: Exception) { /* best effort; backend-opplasting kommer */ }
    }

    /**
     * Analysen forkastet bildet (musingsUI runde 10). Vi trenger ikke noe nytt
     * signal fra CV-kjernen: status != OK ER kvalitetsportens «dette kan jeg ikke
     * score»-svar, så her anbefaler vi rett og slett et nytt bilde.
     */
    private fun renderRejected(status: Int) {
        content.removeAllViews()
        content.addView(Ui.title(this, getString(R.string.app_name)))
        content.addView(Ui.body(this, getString(R.string.rescan_body)))
        content.addView(Ui.hint(this, getString(R.string.result_rejected, status)))
        content.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = getString(R.string.result_send_fail)
            layoutParams = Ui.matchWrap(16, this@ResultActivity)
            setOnClickListener {
                Toast.makeText(this@ResultActivity,
                    getString(R.string.summary_queue, 1), Toast.LENGTH_SHORT).show()
                isEnabled = false
            }
        })
        val btnRow = Ui.row(this)
        btnRow.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = getString(R.string.cancel)
            setOnClickListener { finish() }
        })
        btnRow.addView(android.widget.Space(this), LinearLayout.LayoutParams(0, 1, 1f))
        btnRow.addView(MaterialButton(this).apply {
            text = getString(R.string.rescan_scan)
            minWidth = Ui.dp(this@ResultActivity, 120)
            setOnClickListener {
                startActivity(Intent(this@ResultActivity, CaptureActivity::class.java))
                finish()
            }
        })
        content.addView(btnRow, Ui.matchWrap(12, this))
    }

    private fun render() {
        val r = record ?: return
        content.removeAllViews()

        val mainRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        mainRow.addView(TargetView(this).apply {
            hits = r.shots
            layoutParams = LinearLayout.LayoutParams(0,
                Ui.dp(this@ResultActivity, 300), 1f)
        })
        // Poenglista: OCR-poengene tar presedens og står i SKJERMREKKEFØLGE
        // (musingsUI runde 10). Uten OCR viser vi som før de detekterte i
        // stigende rekkefølge. Blyanten er fjernet — den blokkerte plassen til
        // høyre, og OCR har overtatt rollen som korreksjon.
        val shown = ocrScores ?: r.shots.map { it.decimal }.sorted()
        mainRow.addView(pointsColumn(getString(R.string.result_points_label), shown, 19f))
        content.addView(mainRow)

        // «Poeng:» foran totalen (musingsUI runde 7). Totalen følger det som
        // faktisk vises, slik at OCR-poengene og summen ikke spriker.
        val sumDec = shown.sum()
        val sumInt = shown.sumOf { floor(it).toInt().coerceIn(0, 10) }
        content.addView(TextView(this).apply {
            text = getString(R.string.result_points_prefix, sumDec, sumInt)
            textSize = 28f
        })
        // Dato + tid med god luft mellom (musingsUI runde 7)
        content.addView(Ui.body(this,
            java.time.Instant.ofEpochMilli(r.ts).atZone(java.time.ZoneId.systemDefault())
                .format(DATE_TIME_FMT)))
        val w = store.weapons().firstOrNull { it.id == r.weaponId }
        // «Sittende med anlegg» / «... med reim» (musingsUI runde 6)
        val modText = when (r.modifier) {
            PosModifier.ANLEGG -> " med anlegg"
            PosModifier.REIM -> " med reim"
            else -> ""
        }
        content.addView(Ui.hint(this,
            "${r.position.label}$modText · ${r.distanceM} m · " + (w?.shownName ?: "—")))

        // Mitt gjennomsnitt for denne stillingen: brutt ned på stilling (ikke
        // avstand/hjelpemiddel), sesong + totalt, oppdateres løpende (runde 8).
        addAvgBlock(r)

        if (ocrScores != null && !ocrAgrees) {
            // De to visningene bytter plass ved uenighet (musingsUI runde 10):
            // OCR-poengene står øverst, de identifiserte treffene nederst.
            content.addView(pointsColumn(getString(R.string.result_detected_label),
                r.shots.map { it.decimal }.sorted(), 17f).apply {
                gravity = Gravity.START
                setPadding(0, Ui.dp(this@ResultActivity, 12), 0, 0)
            })
            renderOcrMismatch()
        } else {
            val btnRow = Ui.row(this)
            btnRow.addView(MaterialButton(this, null,
                com.google.android.material.R.attr.borderlessButtonStyle).apply {
                text = getString(R.string.result_discard)
                setOnClickListener { finish() }
            })
            btnRow.addView(android.widget.Space(this),
                LinearLayout.LayoutParams(0, 1, 1f))
            btnRow.addView(MaterialButton(this).apply {
                text = getString(R.string.ok)
                minWidth = Ui.dp(this@ResultActivity, 120)
                setOnClickListener { saveAndFinish() }
            })
            content.addView(btnRow, Ui.matchWrap(12, this))
        }
    }

    /**
     * Poengkolonne med overskrift; verdiene vises i den rekkefølgen de kommer
     * (musingsUI runde 10 — OCR-poengene skal IKKE sorteres på verdi).
     */
    private fun pointsColumn(title: String, values: List<Double>, size: Float): LinearLayout {
        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            // Poengene midtstilles under «Poeng:» (musingsUI runde 9)
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(Ui.dp(this@ResultActivity, 8), 0, 0, 0)
        }
        col.addView(TextView(this).apply {
            text = title
            // Samme størrelse som poengene under, i fet (musingsUI runde 8)
            textSize = size
            gravity = Gravity.CENTER_HORIZONTAL
            setTypeface(typeface, android.graphics.Typeface.BOLD)
        })
        values.forEach { v ->
            col.addView(TextView(this).apply {
                text = "%.1f".format(v)
                textSize = size
                gravity = Gravity.CENTER
                minWidth = Ui.dp(this@ResultActivity, 40)
            }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                topMargin = Ui.dp(this@ResultActivity, 1)
            })
        }
        return col
    }

    /**
     * «Mitt gjennomsnitt for denne stillingen» — sesong + totalt, brutt ned kun
     * på stilling (musingsUI runde 8). Tar med gjeldende serie selv om den ennå
     * ikke er lagret, så snittet stemmer med det som vises.
     */
    private fun addAvgBlock(r: SeriesRecord) {
        val samePos = store.allSeries().filter { it.position == r.position && it.id != r.id } + r
        val seasonNow = Store.seasonKey(r.ts)
        val season = samePos.filter { Store.seasonKey(it.ts) == seasonNow }
        content.addView(TextView(this).apply {
            text = getString(R.string.result_avg_pos_title)
            textSize = 15f
            setTypeface(typeface, android.graphics.Typeface.BOLD)
            setPadding(0, Ui.dp(this@ResultActivity, 8), 0, Ui.dp(this@ResultActivity, 2))
        })
        if (season.isNotEmpty())
            content.addView(Ui.body(this, getString(R.string.result_avg_season,
                season.map { it.sumDecimal }.average())))
        content.addView(Ui.body(this, getString(R.string.result_avg_total,
            samePos.map { it.sumDecimal }.average())))
    }

    /** OCR uenig (> 0.2): vis melding + forkast / lagre de leste poengene. */
    private fun renderOcrMismatch() {
        val ocr = ocrScores ?: return
        content.addView(Ui.body(this, getString(R.string.ocr_mismatch)))
        val btnRow = Ui.row(this)
        btnRow.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = getString(R.string.ocr_discard)
            setOnClickListener { maybeAskDonate { finish() } }
        })
        btnRow.addView(android.widget.Space(this), LinearLayout.LayoutParams(0, 1, 1f))
        btnRow.addView(MaterialButton(this).apply {
            text = getString(R.string.ocr_save_screen)
            setOnClickListener {
                applyScores(ocr)
                ocrAgrees = true
                maybeAskDonate { saveAndFinish() }
            }
        })
        content.addView(btnRow, Ui.matchWrap(12, this))
    }

    /** «Vi vil gjerne bruke dette bildet» — kun hvis dev-deling ikke er aktiv. */
    private fun maybeAskDonate(after: () -> Unit) {
        val r = record
        if (store.shareDevImagesActive || r == null) {
            if (store.shareDevImagesActive) r?.let { queueDevImage(it, "ocr_mismatch") }
            after(); return
        }
        AlertDialog.Builder(this)
            .setTitle(R.string.donate_title)
            .setMessage(R.string.donate_body)
            .setPositiveButton(R.string.donate_accept) { _, _ ->
                store.shareDevImages = "ja"
                queueDevImage(r, "ocr_mismatch")
                after()
            }
            .setNegativeButton(R.string.donate_decline) { _, _ -> after() }
            .setOnCancelListener { after() }
            .show()
    }

    private fun saveAndFinish() {
        val r = record ?: return finish()
        // Varsel kun mot FORRIGE serie (musingsUI runde 5)
        val prev = store.allSeries().filter { it.id != r.id }.maxByOrNull { it.ts }
        if (prev != null && !saved && isIdentical(prev, r)) {
            AlertDialog.Builder(this)
                .setTitle(R.string.dup_title)
                .setMessage(R.string.dup_body)
                .setPositiveButton(R.string.save) { _, _ -> commitAndFinish() }
                .setNegativeButton(R.string.cancel) { _, _ -> finish() }
                .show()
        } else {
            commitAndFinish()
        }
    }

    private fun commitAndFinish() {
        val r = record ?: return finish()
        if (!saved) { store.addSeries(r); saved = true }
        if (r.corrected && !r.sendToFailChannel) {
            Dialogs.failChannelConsent(this) { yes ->
                if (yes) { r.sendToFailChannel = true; store.updateSeries(r) }
                consentFlow()
            }
        } else {
            consentFlow()
        }
    }

    /**
     * Samtykke-rekkefølge (musingsUI runde 6): «Mitt jaktmål» først (etter tre
     * serier), «Bidra til forskning» tidligst to serier senere — aldri begge på
     * samme serie.
     */
    private fun consentFlow() {
        val n = store.currentSeasonSeries().size
        if (!store.jaktmaalPromptSeen && n >= 3) {
            store.jaktmaalPromptSeen = true
            store.jaktmaalPromptCount = n
            Dialogs.jaktmaalDialog(this, store) { finish() }
            return
        }
        Dialogs.maybeResearchConsent(this, store) { finish() }
    }

    /**
     * To serier er «like» bare hvis BÅDE poengene OG treffpunktene stemmer
     * (musingsUI runde 10). Like poeng alene holdt ikke: et opp-ned bilde gir
     * nøyaktig de samme poengene (radiene er uendret) men speilvendte/roterte
     * treffpunkter, og ble derfor feilaktig meldt som duplikat.
     */
    private fun isIdentical(a: SeriesRecord, b: SeriesRecord): Boolean {
        if (a.id == b.id) return false
        if (a.position != b.position || a.distanceM != b.distanceM) return false
        if (a.shots.size != b.shots.size || a.shots.isEmpty()) return false
        val sa = a.shots.map { it.decimal }.sorted()
        val sb = b.shots.map { it.decimal }.sorted()
        if (!sa.indices.all { abs(sa[it] - sb[it]) < 0.05 }) return false
        // Parer treffpunktene grådig: hvert treff i a må ha et ubrukt treff i b
        // innenfor 0,1 poeng (rRel er i ringsteg = poeng).
        val free = b.shots.toMutableList()
        for (s in a.shots) {
            val match = free.minByOrNull { hitDistance(s, it) } ?: return false
            if (hitDistance(s, match) > SAME_HIT_POINTS) return false
            free.remove(match)
        }
        return true
    }

    /** Avstand mellom to treffpunkter, målt i poeng (ringsteg). */
    private fun hitDistance(p: Shot, q: Shot): Double {
        val dx = p.rRel * cos(p.theta) - q.rRel * cos(q.theta)
        val dy = p.rRel * sin(p.theta) - q.rRel * sin(q.theta)
        return sqrt(dx * dx + dy * dy)
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        if (saved) record?.let { outState.putString("recordId", it.id) }
    }
}
