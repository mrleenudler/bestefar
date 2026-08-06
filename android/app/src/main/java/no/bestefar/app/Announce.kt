package no.bestefar.app

import android.app.Activity
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import org.json.JSONObject

/**
 * Kunngjøring av felt dyr til vennene (backend_spec §3).
 *
 * Meldingen er FLYKTIG: serveren sender en push og lagrer ingenting om hva som
 * ble felt eller hvor. Jaktloggen ligger i den klient-krypterte kopien, og den
 * skal den fortsette å ligge i.
 *
 * **Forhåndsvisning er hovedpoenget her** (backendens anbefaling, musingsUI
 * runde 13): brukeren skal se den nøyaktige setningen vennene får, før den
 * sendes. Et delingsvalg der man må gjette hva som deles, er ikke et valg.
 * Derfor bygger klienten den *samme* teksten som serveren, og stedet er et felt
 * brukeren kan rette eller tømme i selve dialogen.
 */
object Announce {

    /**
     * Ubestemt form med artikkel — serveren limer teksten sammen som
     * «{navn} har felt {art}», så bøyningen må komme herfra.
     */
    fun speciesPhrase(r: HuntRecord): String = when (r.species) {
        Species.ELG -> "en elg"
        Species.HJORT -> "en hjort"
        Species.VILLREIN -> "en villrein"
        Species.RAADYR -> "et rådyr"
        Species.VILLSVIN -> "et villsvin"
        Species.ANNET -> r.speciesOther?.takeIf { it.isNotBlank() }?.let { "en $it" } ?: "et dyr"
    }

    /** Kun vellykkede fellinger kunngjøres — ikke bom, ikke ettersøk. */
    fun eligible(r: HuntRecord): Boolean =
        r.fellingSuccess && r.outcome == Outcome.DOEDELIG

    /**
     * Viser forhåndsvisningen og sender hvis brukeren vil. [then] kalles i
     * ALLE tilfeller (også ved avbryt) — kunngjøringen er en bonus på toppen av
     * lagringen, aldri noe som kan stoppe den.
     */
    fun offer(a: Activity, r: HuntRecord, then: () -> Unit) {
        if (!eligible(r)) { then(); return }
        val store = Store.get(a)
        if (!Auth.isLoggedIn(a)) { then(); return }

        val art = speciesPhrase(r)
        val stedFelt = EditText(a).apply {
            setText(r.placeName ?: "")
            hint = a.getString(R.string.announce_place_hint)
            isSingleLine = true
        }
        val preview = TextView(a).apply {
            textSize = 17f
            setPadding(0, Ui.dp(a, 12), 0, Ui.dp(a, 12))
        }
        fun tekst(): String {
            val navn = store.accountName.ifEmpty { a.getString(R.string.announce_you) }
            val sted = stedFelt.text.toString().trim()
            return if (sted.isEmpty()) "$navn har felt $art."
            else "$navn har felt $art i $sted."
        }
        preview.text = "«${tekst()}»"
        stedFelt.addTextChangedListener(object : android.text.TextWatcher {
            override fun afterTextChanged(s: android.text.Editable?) {
                preview.text = "«${tekst()}»"
            }
            override fun beforeTextChanged(s: CharSequence?, a1: Int, b: Int, c: Int) {}
            override fun onTextChanged(s: CharSequence?, a1: Int, b: Int, c: Int) {}
        })

        val col = Ui.col(a, 20)
        col.addView(TextView(a).apply { setText(R.string.announce_body) })
        col.addView(preview)
        col.addView(TextView(a).apply {
            setText(R.string.announce_place_label)
            textSize = 13f
        })
        col.addView(stedFelt)
        col.addView(Ui.hint(a, a.getString(R.string.announce_note)))

        AlertDialog.Builder(a)
            .setTitle(R.string.announce_title)
            .setView(androidx.core.widget.NestedScrollView(a).apply { addView(col) })
            .setPositiveButton(R.string.announce_send) { _, _ ->
                send(a, art, stedFelt.text.toString().trim())
                then()
            }
            .setNegativeButton(R.string.announce_skip) { _, _ -> then() }
            .setOnCancelListener { then() }
            .show()
    }

    private fun send(a: Activity, art: String, sted: String) {
        Api.io {
            val body = JSONObject().put("species", art)
            if (sted.isNotEmpty()) body.put("kommune", sted)
            val resp = Api.postJson(a, "/v1/hunts/announce", body)
            Api.ui {
                if (a.isFinishing) return@ui
                Ui.toast(a, when {
                    // Serveren svarer med antall ENHETER som fikk varselet.
                    // Vi lover ikke brukeren mer enn det som faktisk skjedde.
                    resp.ok -> {
                        val n = try { JSONObject(resp.body).optInt("devices_notified", 0) }
                                catch (_: Exception) { 0 }
                        if (n > 0) a.getString(R.string.announce_sent, n)
                        else a.getString(R.string.announce_sent_none)
                    }
                    resp.code == 401 -> a.getString(R.string.announce_need_login)
                    resp.code == 403 -> a.getString(R.string.announce_sharing_off)
                    resp.code == 429 -> a.getString(R.string.announce_too_soon)
                    resp.code == 0 -> a.getString(R.string.announce_offline)
                    else -> a.getString(R.string.announce_failed, resp.code)
                })
            }
        }
    }
}
