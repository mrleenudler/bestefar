package no.bestefar.app

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.time.Instant
import java.time.LocalDateTime
import java.time.OffsetDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Locale

/**
 * Meldingskoeen (backend_spec §11).
 *
 * **Koeen er garantien, pushen er bare rask levering.** Backenden legger
 * §11-varsler i koeen FOER pushen sendes, og en push som ikke kommer fram - fordi
 * brukeren har avslaatt varsler, fordi FCM-tokenet er rotert, fordi telefonen var
 * av - er ikke et tapt varsel saa lenge noen henter koeen. Fram til v0.19 gjorde
 * ingen det, og garantien fantes bare paa papiret.
 *
 * Meldingene det gjelder taaler ikke aa forsvinne: «Lagleder har fjernet deg fra
 * {lag}», «{navn} vil gjoere deg til lagleder - bekreft i appen», «{lag} skal
 * velge lagleder. Avstemningen er aapen i 7 dager». En frist som gaar ut fordi
 * meldingen aldri ble vist, kan ikke rettes opp etterpaa.
 *
 * **Skjemaet er lest ut av `backend/app/routers/messages.py` og
 * `models/social.py`, ikke ut av kontrakten** - `backend/KONTRAKT.md` har ingen
 * seksjon for koeen (meldt som issue #5). To ting er verdt aa merke seg, fordi de
 * bryter med moensteret ellers i API-et:
 *
 *  - `id` er et **heltall**, mens `team_id` er en **streng**-UUID eller `null`.
 *  - `kind` er en fri streng paa 32 tegn, **ikke** et enum. Derfor lagres den
 *    som `String` her: en `enum class` ville krasjet den dagen backenden legger
 *    til en aattende meldingstype, som er nettopp det som skal kunne skje uten
 *    en ny klientutgivelse.
 *
 * Alt nettverksarbeid gaar via [Api.io]; callbackene kommer paa UI-traaden.
 */
object Messages {

    private const val TAG = "BestefarMsg"
    private const val PATH = "/v1/messages"

    private val fmt = DateTimeFormatter.ofPattern("d. MMMM yyyy 'kl.' HH:mm",
        Locale.forLanguageTag("no"))

    /**
     * En ventende melding. [created_at] beholdes som den ferdig formaterte
     * teksten - klienten gjoer ingenting annet med tidspunktet enn aa vise det,
     * og en tom streng er riktigere enn en ISO-streng brukeren ikke kan lese.
     */
    class Msg(val id: Int, val kind: String, val title: String,
              val body: String, val teamId: String?, val naar: String)

    // ---------- Henting ----------

    /**
     * Henter uleverte meldinger, eldste foerst. Tom liste ved alt som kan gaa
     * galt: ingen konto, offline, 401, ubrukelig svar. Dette er en oppstartsjobb
     * i en offline-foerst app - den skal aldri vise en feil, og den skal aldri
     * hindre at appen starter. Koeen ligger igjen paa serveren og hentes ved
     * neste oppstart.
     */
    fun fetch(ctx: Context, onDone: (List<Msg>) -> Unit) {
        if (!Auth.isLoggedIn(ctx)) { Api.ui { onDone(emptyList()) }; return }
        Api.io {
            val resp = Api.send(ctx, "GET", PATH)
            val list = if (resp.ok) parse(resp.body) else {
                if (resp.code != 0) Log.d(TAG, "Koeen kunne ikke hentes (${resp.code})")
                emptyList()
            }
            Api.ui { onDone(list) }
        }
    }

    private fun parse(body: String): List<Msg> = try {
        val arr = JSONArray(body)
        (0 until arr.length()).mapNotNull { i ->
            val o = arr.optJSONObject(i) ?: return@mapNotNull null
            val id = o.optInt("id", 0)
            val tekst = o.optString("body", "")
            // Uten ID kan vi ikke kvittere, og uten broedtekst er det ingenting
            // aa vise. Hopp over raden framfor aa vise et tomt vindu brukeren
            // maa klikke bort.
            if (id == 0 || tekst.isBlank()) return@mapNotNull null
            Msg(
                id = id,
                kind = o.optString("kind", ""),
                title = o.optString("title", ""),
                body = tekst,
                // isNull() foerst: org.json lar `optString` gi strengen "null"
                // for en JSON-null, og `team_id` ER null for meldinger som ikke
                // gjelder et lag.
                teamId = if (o.isNull("team_id")) null
                         else o.optString("team_id", "").takeIf { it.isNotEmpty() },
                naar = formatNaar(o.optString("created_at", "")))
        }
    } catch (e: Exception) {
        Log.w(TAG, "Ubrukelig svar fra $PATH", e)
        emptyList()
    }

    /**
     * Serveren sender ISO-8601. Den kan komme med offset (`+00:00`), med `Z`,
     * eller naken - derfor tre forsoek foer vi gir opp. Feiler alle, vises
     * ingen dato; en raa ISO-streng i et varselvindu er stoey, ikke informasjon.
     */
    private fun formatNaar(iso: String): String {
        if (iso.isBlank()) return ""
        val zone = ZoneId.systemDefault()
        val tid = try {
            OffsetDateTime.parse(iso).atZoneSameInstant(zone).toLocalDateTime()
        } catch (_: Exception) {
            try {
                Instant.parse(iso).atZone(zone).toLocalDateTime()
            } catch (_: Exception) {
                try { LocalDateTime.parse(iso) } catch (_: Exception) { null }
            }
        }
        return tid?.format(fmt) ?: ""
    }

    // ---------- Kvittering ----------

    /**
     * Kvitterer for meldinger som **er vist**. Kalles aldri ved henting: serveren
     * markerer raden med `delivered_at` i stedet for aa slette den nettopp for aa
     * taale en klient som forsvinner mellom henting og visning, og den
     * toleransen er verdiloes hvis vi kvitterer for tidlig.
     *
     * Feiler kallet, gjoer vi ingenting. Meldingen kommer igjen ved neste
     * oppstart, og aa se den samme beskjeden to ganger er en billigere feil enn
     * aldri aa se den.
     */
    fun ack(ctx: Context, ids: List<Int>) {
        if (ids.isEmpty()) return
        Api.io {
            val body = JSONObject().put("ids", JSONArray().apply { ids.forEach { put(it) } })
            val resp = Api.postJson(ctx, "$PATH/ack", body)
            if (!resp.ok) Log.d(TAG, "Kvittering feilet (${resp.code}) for $ids")
        }
    }

    fun ack(ctx: Context, id: Int) = ack(ctx, listOf(id))
}
