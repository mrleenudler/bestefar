package no.bestefar.app

import android.content.Intent
import android.text.Editable
import android.text.TextWatcher
import android.widget.EditText
import android.widget.LinearLayout
import androidx.appcompat.app.AlertDialog
import com.google.android.material.button.MaterialButton

/**
 * Meny (musingsUI): rediger/legg til våpen (m/optikk og ammo), serielogg,
 * optikk-kalkulator, søk, profil, historikk, mer statistikk, hjelp, tutorial.
 */
class MenyFragment : RebuildFragment() {

    override fun rebuild() {
        val a = requireActivity()
        val store = Store.get(a)
        content.removeAllViews()
        content.addView(Ui.title(a, getString(R.string.tab_meny)))

        fun entry(label: String, onClick: () -> Unit) {
            content.addView(MaterialButton(a, null,
                com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
                text = label
                layoutParams = Ui.matchWrap(8, a).apply { height = Ui.dp(a, 56) }
                setOnClickListener { onClick() }
            })
        }

        entry(getString(R.string.search)) { searchDialog() }

        entry(getString(R.string.weapon_edit)) {
            val ws = store.weapons()
            if (ws.isEmpty()) {
                Dialogs.weaponEdit(a, store, null, true) { rebuild() }
            } else {
                AlertDialog.Builder(a)
                    .setTitle(R.string.weapon_edit)
                    .setItems(ws.map { it.shownName }.toTypedArray()) { _, i ->
                        Dialogs.weaponEdit(a, store, ws[i], true) { rebuild() }
                    }.show()
            }
        }

        entry(getString(R.string.weapon_add)) {
            Dialogs.weaponEdit(a, store, null, true) { rebuild() }
        }

        entry(getString(R.string.serielogg)) {
            startActivity(Intent(a, SerieloggActivity::class.java))
        }

        entry(getString(R.string.kalkulator)) {
            startActivity(Intent(a, KalkulatorActivity::class.java))
        }

        entry(getString(R.string.menu_profile)) {
            startActivity(Intent(a, ProfilActivity::class.java))
        }

        entry(getString(R.string.menu_history)) { historyDialog() }

        entry(getString(R.string.menu_stats)) { statsDialog() }

        entry(getString(R.string.menu_help)) {
            AlertDialog.Builder(a).setTitle(R.string.menu_help)
                .setMessage(R.string.help_body)
                .setPositiveButton(R.string.ok, null).show()
        }

        entry(getString(R.string.menu_tutorial)) {
            (a as? MainActivity)?.let { it.showHome(); it.showTutorial() }
        }
    }

    private fun historyDialog() {
        val a = requireActivity()
        val store = Store.get(a)
        // Historikk (spec §5): tidligere sesonger; frosne kart kommer
        val bySeason = store.allSeries().groupBy { Store.seasonKey(it.ts) }
            .toSortedMap(compareByDescending { it })
        val msg = if (bySeason.isEmpty()) "Ingen serier ennå."
        else bySeason.entries.joinToString("\n") { (k, list) ->
            val avg = list.map { it.sumDecimal }.average()
            "${Store.seasonLabel(k)}: ${list.size} serier, " +
                "${Stats.shotCount(list)} skudd, snitt %.1f".format(avg)
        } + "\n\nFrosne sesongkart og utviklingskurver kommer."
        AlertDialog.Builder(a).setTitle(R.string.menu_history)
            .setMessage(msg).setPositiveButton(R.string.ok, null).show()
    }

    private fun statsDialog() {
        val a = requireActivity()
        val store = Store.get(a)
        // Tekniske mål KUN her og i forskningseksport (spec §1)
        val evid = store.currentSeasonSeries().filter { it.countsInEvidence }
        val sigma = Stats.sigmaCmAt100(evid)
        val msg = if (sigma == null) "Ingen data denne sesongen."
        else "σ (100 m-ekvivalent): %.1f cm\nR95: %.1f cm\nSpredning: %.2f MOA\n\nMålt om siktepunktet (%d skudd). Benk holdes utenfor."
            .format(sigma, Stats.r95Cm(sigma), Stats.moa(sigma), Stats.shotCount(evid))
        AlertDialog.Builder(a).setTitle(R.string.menu_stats)
            .setMessage(msg).setPositiveButton(R.string.ok, null).show()
    }

    // ---------- Søk (musingsUI): fritekst mot funksjons-/info-indeks ----------

    private data class SearchEntry(val title: String, val keywords: String,
                                   val action: () -> Unit)

    private fun searchIndex(): List<SearchEntry> {
        val a = requireActivity()
        val main = a as? MainActivity
        return listOf(
            SearchEntry(getString(R.string.serielogg),
                "serie logg historikk resultat slette gjennomsnitt") {
                startActivity(Intent(a, SerieloggActivity::class.java)) },
            SearchEntry(getString(R.string.kalkulator),
                "optikk kalkulator moa mrad cm klikk omregning smoa") {
                startActivity(Intent(a, KalkulatorActivity::class.java)) },
            SearchEntry(getString(R.string.weapon_edit),
                "våpen rediger optikk ammo ammunisjon kikkert klikkverdi") {
                Dialogs.weaponEdit(a, Store.get(a), Store.get(a).weapons().firstOrNull(),
                    true) {} },
            SearchEntry(getString(R.string.weapon_add),
                "våpen legg til ny rifle") {
                Dialogs.weaponEdit(a, Store.get(a), null, true) {} },
            SearchEntry(getString(R.string.menu_profile),
                "profil samtykke skadeskytingsrate sletting data fødselsår lag") {
                startActivity(Intent(a, ProfilActivity::class.java)) },
            SearchEntry(getString(R.string.tab_innsikt),
                "innsikt kompetanse kart forsvarlig hold frekvens") {
                main?.select(4) },
            SearchEntry(getString(R.string.tab_jakt),
                "jakt hurtiglogg skudd art hold vinkling utfall ettersøk") {
                main?.select(2) },
            SearchEntry(getString(R.string.tab_avstand),
                "avstand meter hold 100") { main?.select(1) },
            SearchEntry(getString(R.string.tab_stilling),
                "stilling liggende sittende knestående stående benk anlegg reim") {
                main?.select(3) },
            SearchEntry(getString(R.string.menu_history),
                "historikk sesong jaktår utvikling") { historyDialog() },
            SearchEntry(getString(R.string.menu_stats),
                "statistikk sigma r95 moa spredning teknisk") { statsDialog() },
            SearchEntry(getString(R.string.menu_help),
                "hjelp ikon forklaring skadeskytingsrate forskning kilder") {
                AlertDialog.Builder(a).setTitle(R.string.menu_help)
                    .setMessage(R.string.help_body)
                    .setPositiveButton(R.string.ok, null).show() },
            SearchEntry(getString(R.string.menu_tutorial),
                "tutorial gjennomgang intro opplæring") {
                main?.let { it.showHome(); it.showTutorial() } },
        )
    }

    private fun searchDialog() {
        val a = requireActivity()
        val root = LinearLayout(a).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(Ui.dp(a, 20), Ui.dp(a, 8), Ui.dp(a, 20), Ui.dp(a, 8))
        }
        val input = EditText(a).apply { hint = getString(R.string.search_hint) }
        val results = LinearLayout(a).apply { orientation = LinearLayout.VERTICAL }
        root.addView(input); root.addView(results)
        val dialog = AlertDialog.Builder(a)
            .setTitle(R.string.search)
            .setView(root)
            .setNegativeButton(R.string.cancel, null)
            .create()
        val index = searchIndex()
        fun refresh(q: String) {
            results.removeAllViews()
            val terms = q.lowercase().split(" ").filter { it.isNotBlank() }
            val hits = if (terms.isEmpty()) emptyList()
            else index.filter { e ->
                terms.all { t ->
                    e.title.lowercase().contains(t) || e.keywords.contains(t)
                }
            }
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
