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
            // Knappesenter i nedre halvdel, flyttet 10 % opp (musingsUI-runde 2)
            FrameLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, h,
                Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL).apply {
                bottomMargin =
                    (resources.displayMetrics.heightPixels * 0.35).toInt() - h / 2
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

        // Hovedskjermen holdes ren (musingsUI runde 6): kun antall øvelsesskudd
        // (Scan-knappen ligger som eget overlegg nederst).
        content.addView(android.widget.TextView(a).apply {
            text = getString(R.string.okt_shots_season, store.shotsThisSeason())
            textSize = 20f
            setPadding(0, 0, 0, Ui.dp(a, 8))
        })
    }
}
