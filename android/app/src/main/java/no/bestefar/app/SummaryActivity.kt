package no.bestefar.app

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton

/**
 * Øktoppsummering (spec §2): dagens serier med stilling, poeng og
 * frekvensbudskap; usendte opplastinger i kø. Ingen per-serie-dom (spec §5).
 */
class SummaryActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val store = Store.get(this)
        val content = Ui.col(this)
        val scroller = Ui.scroll(this, content)
        Ui.applyInsets(scroller)
        setContentView(scroller)

        content.addView(Ui.title(this, getString(R.string.summary_title)))

        val today = store.seriesToday().sortedBy { it.ts }
        today.forEachIndexed { i, s ->
            val freq = if (s.countsInEvidence) {
                val sigma = Stats.sigmaCmAt100(listOf(s))
                val radius = Stats.lethalRadiusCm(Species.ELG, Angle.SIDE)
                if (sigma != null && radius != null) {
                    val p = Stats.pLethal(sigma, s.distanceM, radius)
                    " · feller rent ${Stats.freqText(p)} (elg, ${s.distanceM} m)"
                } else ""
            } else " · ${getString(R.string.result_counts_not).lowercase()}"
            content.addView(Ui.body(this,
                "${i + 1}. ${s.position.label} · %.1f (%d skudd)$freq"
                    .format(s.sumDecimal, s.shots.size)))
        }

        content.addView(Ui.vspace(this, 8))
        content.addView(Ui.hint(this, getString(R.string.summary_no_change)))

        val queued = store.allSeries().count { !it.uploaded } +
            store.allHunts().size
        content.addView(Ui.body(this, getString(R.string.summary_queue, queued)))

        content.addView(MaterialButton(this).apply {
            text = getString(R.string.ok)
            layoutParams = Ui.matchWrap(16, this@SummaryActivity)
            setOnClickListener { finish() }
        })
    }
}
