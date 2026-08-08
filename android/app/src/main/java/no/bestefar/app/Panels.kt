package no.bestefar.app

import android.content.Intent
import android.text.Editable
import android.text.InputFilter
import android.text.InputType
import android.text.TextWatcher
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.EditText
import android.widget.LinearLayout
import androidx.appcompat.app.AlertDialog
import com.google.android.material.button.MaterialButton

/**
 * Dropdown-paneler (musingsUI runde 4): kun Avstand og Meny åpner som
 * dropdown; Innsikt er fullskjerm. Våpen/Jakt/Stilling er ikke lenger i baren.
 */
object Panels {

    fun build(i: Int, a: MainActivity, refresh: () -> Unit): View = when (i) {
        MainActivity.TAB_AVSTAND -> avstand(a, refresh)
        else -> meny(a)
    }

    // ---------- Avstand: vertikale valg + «X m»-knapp (musingsUI) ----------

    private fun avstand(a: MainActivity, refresh: () -> Unit): View {
        val store = a.store
        val col = Ui.col(a, 12)

        listOf(50, 100, 150, 200, 300).forEach { d ->
            col.addView(Ui.choiceButton(a, "$d m", store.distanceM == d) {
                store.distanceM = d; refresh(); a.closeDropdownDelayed()
            }.apply { layoutParams = Ui.matchWrap(4, a) })
        }

        val custom = store.customDistance
        val customLabel = if (custom > 0) "$custom m" else a.getString(R.string.distance_x)
        val inputRow = Ui.row(a).apply { visibility = View.GONE }
        val input = EditText(a).apply {
            hint = a.getString(R.string.distance_custom_hint)
            inputType = InputType.TYPE_CLASS_NUMBER
            filters = arrayOf(InputFilter.LengthFilter(4))
            layoutParams = LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }
        inputRow.addView(input)
        inputRow.addView(MaterialButton(a).apply {
            text = a.getString(R.string.distance_set)
            setOnClickListener {
                val v = input.text.toString().toIntOrNull()
                when {
                    v != null && v in 10..9999 -> {
                        store.customDistance = v; store.distanceM = v; a.closeDropdown()
                    }
                    // Sett uten ny verdi -> behold gammel og lukk (musingsUI)
                    custom > 0 -> { store.distanceM = custom; a.closeDropdown() }
                }
            }
        })

        var editing = false
        col.addView(Ui.choiceButton(a, customLabel,
            custom > 0 && store.distanceM == custom) {
            when {
                custom <= 0 -> { editing = true; inputRow.visibility = View.VISIBLE }
                store.distanceM != custom -> {
                    store.distanceM = custom; refresh(); a.closeDropdownDelayed()
                }
                else -> {
                    editing = !editing
                    inputRow.visibility = if (editing) View.VISIBLE else View.GONE
                }
            }
        }.apply { layoutParams = Ui.matchWrap(4, a) })
        col.addView(inputRow)
        return col
    }

    // ---------- Meny (musingsUI runde 4-rekkefølge) ----------

    private fun meny(a: MainActivity): View {
        val col = Ui.col(a, 12)
        fun entry(label: String, launchesActivity: Boolean = false, onClick: () -> Unit) {
            col.addView(MaterialButton(a, null,
                com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
                text = label
                textSize = 14f
                layoutParams = Ui.matchWrap(4, a).apply { height = Ui.dp(a, 44) }
                setOnClickListener {
                    if (launchesActivity) a.reopenMenyOnResume = true else a.closeDropdown()
                    onClick()
                }
            })
        }
        entry(a.getString(R.string.menu_profile), launchesActivity = true) {
            a.startActivity(Intent(a, ProfilActivity::class.java))
        }
        entry(a.getString(R.string.menu_jakt), launchesActivity = true) {
            a.startActivity(Intent(a, JaktActivity::class.java))
        }
        entry(a.getString(R.string.menu_friends), launchesActivity = true) {
            a.startActivity(Intent(a, VennerActivity::class.java))
        }
        entry(a.getString(R.string.menu_series), launchesActivity = true) {
            a.startActivity(Intent(a, SerieloggActivity::class.java))
        }
        entry(a.getString(R.string.menu_send_message)) { sendMessageDialog(a) }
        entry(a.getString(R.string.menu_tutorial)) { a.showHome(); a.showTutorial() }
        entry(a.getString(R.string.search)) { searchDialog(a) }
        return col
    }

    /**
     * «Gi tilbakemelding til utvikler» → `POST /v1/feedback` (backend_spec §10).
     * Endepunktet krever ikke innlogging, så dette er den ene funksjonen som
     * fungerer ende-til-ende allerede. Feiler den, faller vi tilbake på
     * e-postappen slik at meldingen aldri går tapt fordi nettet er nede.
     */
    private fun sendMessageDialog(a: MainActivity) {
        val root = Ui.col(a, 16)
        // Grensene er serverens (contracts/openapi.json, FeedbackIn: subject
        // 200, body 10000). Haandhevet i FELTET, ikke ved avkorting foer
        // sending: en bruker som skriver mer enn det skal merke det mens de
        // skriver, ikke miste slutten uten beskjed. Uten grensen svarer
        // serveren 422, og 422 er ikke 429 - da falt vi ut i mailto-grenen og
        // «send melding» aapnet e-postappen uten at noen skjoente hvorfor.
        val title = EditText(a).apply {
            hint = a.getString(R.string.message_title_hint)
            filters = arrayOf(android.text.InputFilter.LengthFilter(200))
        }
        Ui.capitalize(title)
        val body = EditText(a).apply {
            hint = a.getString(R.string.message_body_hint); minLines = 4; gravity = Gravity.TOP
            filters = arrayOf(android.text.InputFilter.LengthFilter(10_000))
        }
        Ui.capitalize(body)
        root.addView(title); root.addView(body)
        AlertDialog.Builder(a)
            .setTitle(R.string.menu_send_message)
            .setView(root)
            .setPositiveButton(R.string.send) { _, _ ->
                val subj = title.text.toString().trim().ifEmpty {
                    a.getString(R.string.menu_send_message)
                }
                val text = body.text.toString().trim()
                if (text.isEmpty()) { Ui.toast(a, R.string.message_empty); return@setPositiveButton }
                sendFeedback(a, subj, text)
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun sendFeedback(a: MainActivity, subject: String, body: String) {
        Ui.toast(a, R.string.message_sending)
        Api.io {
            val resp = Api.postJson(a, "/v1/feedback", org.json.JSONObject().apply {
                put("subject", subject)
                put("body", body)
                put("app_version", BuildConfig.VERSION_NAME)
                // take(64): skjemaets grense. Uten den ville en telefon med et
                // langt fabrikant- og modellnavn faatt 422 paa HELE
                // tilbakemeldingen, av en grunn som ikke har noe med meldingen
                // aa gjoere. Push.kt avkorter allerede likt.
                put("device_model",
                    "${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}".take(64))
            })
            Api.ui {
                when {
                    resp.ok -> Ui.toast(a, R.string.message_sent)
                    // 429: serveren ber oss vente. Da hjelper det ikke aa aapne
                    // e-postappen heller — meldingen er allerede sendt inn.
                    resp.code == 429 -> Ui.toast(a, R.string.message_rate_limited)
                    else -> mailtoFallback(a, subject, body)
                }
            }
        }
    }

    /** Legg subject/body i selve mailto-URIen — mer pålitelig enn EXTRA_* (runde 6). */
    private fun mailtoFallback(a: MainActivity, subject: String, body: String) {
        val subj = android.net.Uri.encode("[Bestefar] $subject")
        val bodyEnc = android.net.Uri.encode(body)
        val intent = Intent(Intent.ACTION_SENDTO).apply {
            data = android.net.Uri.parse("mailto:mrleenudler@gmail.com?subject=$subj&body=$bodyEnc")
        }
        try {
            Ui.toast(a, R.string.message_offline_email)
            a.startActivity(intent)
        } catch (_: Exception) {
            Ui.toast(a, R.string.message_no_email_app)
        }
    }

    // ---------- Søk ----------

    private data class SearchEntry(val title: String, val keywords: String,
                                   val action: () -> Unit)

    private fun searchIndex(a: MainActivity): List<SearchEntry> {
        val store = a.store
        return listOf(
            SearchEntry(a.getString(R.string.menu_series),
                "serie serier logg resultat slette gjennomsnitt sesong") {
                a.startActivity(Intent(a, SerieloggActivity::class.java)) },
            SearchEntry(a.getString(R.string.menu_profile),
                "profil kallenavn visningsnavn jaktmål skadeskudd sletting fødselsår lag tema modus") {
                a.startActivity(Intent(a, ProfilActivity::class.java)) },
            SearchEntry(a.getString(R.string.menu_jakt),
                "jakt jaktlogg registrer skudd art hold vinkling utfall ettersøk vilt") {
                a.startActivity(Intent(a, JaktActivity::class.java)) },
            SearchEntry(a.getString(R.string.menu_friends),
                "venner venn legg til qr brukerid deling lag skytterlag") {
                a.startActivity(Intent(a, VennerActivity::class.java)) },
            SearchEntry(a.getString(R.string.tab_innsikt),
                "innsikt kompetanse kart forsvarlig hold frekvens statistikk mer") {
                a.select(MainActivity.TAB_INNSIKT) },
            SearchEntry(a.getString(R.string.tab_avstand),
                "avstand meter hold egendefinert") { a.select(MainActivity.TAB_AVSTAND) },
            SearchEntry(a.getString(R.string.menu_send_message),
                "tilbakemelding melding feil problem epost kontakt utvikler") {
                sendMessageDialog(a) },
            SearchEntry(a.getString(R.string.menu_tutorial),
                "hvordan bruke appen tutorial gjennomgang opplæring velkommen") {
                a.showHome(); a.showTutorial() },
        )
    }

    private fun searchDialog(a: MainActivity) {
        val root = LinearLayout(a).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(Ui.dp(a, 20), Ui.dp(a, 8), Ui.dp(a, 20), Ui.dp(a, 8))
        }
        val input = EditText(a).apply { hint = a.getString(R.string.search_hint) }
        Ui.capitalize(input)
        val results = LinearLayout(a).apply { orientation = LinearLayout.VERTICAL }
        root.addView(input); root.addView(results)
        val dialog = AlertDialog.Builder(a)
            .setTitle(R.string.search).setView(root)
            .setNegativeButton(R.string.cancel, null).create()
        val index = searchIndex(a)
        fun refresh(q: String) {
            results.removeAllViews()
            val terms = q.lowercase().split(" ").filter { it.isNotBlank() }
            val hits = if (terms.isEmpty()) emptyList()
            else index.filter { e -> terms.all { t ->
                e.title.lowercase().contains(t) || e.keywords.contains(t) } }
            hits.take(6).forEach { e ->
                results.addView(MaterialButton(a, null,
                    com.google.android.material.R.attr.borderlessButtonStyle).apply {
                    text = e.title
                    setOnClickListener { dialog.dismiss(); e.action() }
                })
            }
        }
        input.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, x: Int, y: Int, z: Int) {}
            override fun onTextChanged(s: CharSequence?, x: Int, y: Int, z: Int) {}
            override fun afterTextChanged(s: Editable?) = refresh(s?.toString() ?: "")
        })
        dialog.show()
    }
}
