package no.bestefar.app

import android.content.Intent
import androidx.appcompat.app.AlertDialog
import com.google.android.material.button.MaterialButton

/** Meny-fanen: Profil (samtykke/datastyring), historikk, mer statistikk, hjelp. */
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

        entry(getString(R.string.menu_profile)) {
            startActivity(Intent(a, ProfilActivity::class.java))
        }

        entry(getString(R.string.menu_history)) {
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

        entry(getString(R.string.menu_stats)) {
            // Tekniske mål KUN her og i forskningseksport (spec §1)
            val evid = store.currentSeasonSeries().filter { it.countsInEvidence }
            val sigma = Stats.sigmaCmAt100(evid)
            val msg = if (sigma == null) "Ingen data denne sesongen."
            else "σ (100 m-ekvivalent): %.1f cm\nR95: %.1f cm\nSpredning: %.2f MOA\n\nMålt om siktepunktet (%d skudd). Benk holdes utenfor."
                .format(sigma, Stats.r95Cm(sigma), Stats.moa(sigma), Stats.shotCount(evid))
            AlertDialog.Builder(a).setTitle(R.string.menu_stats)
                .setMessage(msg).setPositiveButton(R.string.ok, null).show()
        }

        entry(getString(R.string.menu_help)) {
            AlertDialog.Builder(a).setTitle(R.string.menu_help)
                .setMessage(R.string.help_body)
                .setPositiveButton(R.string.ok, null).show()
        }
    }
}
