package no.bestefar.app

import android.content.Intent
import android.content.res.Configuration
import android.os.Bundle
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.LinearLayout
import androidx.fragment.app.Fragment
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter

/**
 * Økt-flaten (hjem): kontekstlinje, øvelsesforslag og dagens serier øverst;
 * stor scan-knapp — stående: sentrert på nedre halvdel; liggende: full
 * bredde nederst (musingsUI). Ingen overskrift.
 */
class OktFragment : Fragment() {

    companion object {
        /** «Ikke nå» demper forslaget for resten av prosess-økta — ikke masete. */
        var suggestionDismissed = false
    }

    private lateinit var content: LinearLayout

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?,
                              savedInstanceState: Bundle?): View {
        val a = requireActivity()
        val root = FrameLayout(a)
        content = Ui.col(a)
        root.addView(Ui.scroll(a, content), ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT)

        val landscape =
            resources.configuration.orientation == Configuration.ORIENTATION_LANDSCAPE
        val scan = MaterialButton(a).apply {
            text = getString(R.string.scan_series)
            textSize = 22f
            setOnClickListener { startActivity(Intent(a, CaptureActivity::class.java)) }
        }
        val h = Ui.dp(a, 84)   // ~50 % større enn forrige knapp
        val lp = if (landscape) {
            FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, h,
                Gravity.BOTTOM).apply {
                leftMargin = Ui.dp(a, 8); rightMargin = Ui.dp(a, 8)
                bottomMargin = Ui.dp(a, 8)
            }
        } else {
            // Sentrert på nedre halvdel: knappesenter ~3/4 ned på flaten
            FrameLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, h,
                Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL).apply {
                bottomMargin = resources.displayMetrics.heightPixels / 4 - h / 2
            }
        }
        if (!landscape) scan.minWidth = Ui.dp(a, 280)
        root.addView(scan, lp)
        return root
    }

    override fun onResume() {
        super.onResume()
        rebuild()
    }

    private fun rebuild() {
        val a = requireActivity()
        val store = Store.get(a)
        content.removeAllViews()

        val weaponName = store.selectedWeapon()?.shownName ?: "—"
        val posText = store.practicePosition?.let { "Øvelse: ${it.label}" }
            ?: "${store.currentPosition.label} (${store.currentModifier.label})"
        content.addView(Ui.body(a,
            getString(R.string.okt_context, weaponName, store.distanceM, posText)))

        // Øvelsesmotoren: popup-forslag som kort, bare ved reelt behov (spec §5)
        val sug = Stats.practiceSuggestion(store.currentSeasonSeries())
        if (sug != null && store.practicePosition == null && !suggestionDismissed) {
            val (pos, trained, prior) = sug
            val card = MaterialCardView(a).apply {
                radius = Ui.dp(a, 12).toFloat()
                layoutParams = Ui.matchWrap(12, a)
            }
            val inner = Ui.col(a, 16)
            inner.addView(Ui.body(a,
                "Du har trent ${(trained * 100).toInt()} % ${pos.label.lowercase()}, " +
                "men i jakt står den for ~${(prior * 100).toInt()} % av skuddene. " +
                "Test ${pos.label.lowercase()} på 100 m?"))
            val btnRow = Ui.row(a)
            btnRow.addView(MaterialButton(a).apply {
                text = getString(R.string.okt_practice_ok)
                setOnClickListener {
                    store.practicePosition = pos
                    store.distanceM = 100
                    rebuild()
                }
            })
            btnRow.addView(MaterialButton(a, null,
                com.google.android.material.R.attr.borderlessButtonStyle).apply {
                text = getString(R.string.okt_practice_dismiss)
                setOnClickListener { suggestionDismissed = true; rebuild() }
            })
            inner.addView(btnRow)
            card.addView(inner)
            content.addView(card)
        }

        val today = store.seriesToday()
        if (today.isEmpty()) {
            content.addView(Ui.hint(a, getString(R.string.okt_no_series)))
        } else {
            val fmt = DateTimeFormatter.ofPattern("HH:mm")
            today.sortedByDescending { it.ts }.forEach { s ->
                val t = Instant.ofEpochMilli(s.ts).atZone(ZoneId.systemDefault()).format(fmt)
                val mod = if (s.modifier != PosModifier.UTEN) " (${s.modifier.label})" else ""
                content.addView(Ui.body(a,
                    "$t · ${s.position.label}$mod · %.1f (%d skudd)"
                        .format(s.sumDecimal, s.shots.size)))
            }
            content.addView(MaterialButton(a, null,
                com.google.android.material.R.attr.borderlessButtonStyle).apply {
                text = getString(R.string.summary_title)
                setOnClickListener { startActivity(Intent(a, SummaryActivity::class.java)) }
            })
        }
    }
}
