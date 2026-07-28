package no.bestefar.app

import android.content.res.Configuration
import android.graphics.Color
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.view.Gravity
import android.view.ViewGroup
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import com.google.android.material.button.MaterialButton
import kotlin.math.roundToInt

/** Felles skjelett for programmatiske fragmenter. */
abstract class RebuildFragment : androidx.fragment.app.Fragment() {
    protected lateinit var content: LinearLayout

    override fun onCreateView(inflater: android.view.LayoutInflater,
                              container: ViewGroup?,
                              savedInstanceState: android.os.Bundle?): android.view.View {
        content = Ui.col(requireContext())
        return Ui.scroll(requireContext(), content)
    }

    override fun onResume() { super.onResume(); rebuild() }
    protected abstract fun rebuild()
}

/**
 * Innsikt (musingsUI runde 5): matrise der de fem vilttypene (rader) rammes
 * inn av innstillingene — jeger-stilling øverst, dyr-vinkling nederst, skuddhold
 * i kolonne til høyre. Vilt-silhuett skalerer med hold; frekvenstekst er grønn
 * når jaktmålet er nådd, rød ellers, og grå «øv på stillingen» der stilling
 * mangler øvelsesskudd. Grønn tekst = mål nådd (jf. (i)).
 */
class InnsiktFragment : RebuildFragment() {

    private var jegerPos = Position.SITTENDE          // forhåndsvalgt
    private var dyrAngle = Angle.SIDE                 // bredside forhåndsvalgt
    private var holdM = 100                           // forhåndsvalgt
    private val holds = listOf(25, 50, 75, 100, 150, 200)
    private val species = listOf(Species.ELG, Species.HJORT, Species.VILLREIN,
        Species.RAADYR, Species.VILLSVIN)

    private val greenCol = Color.parseColor("#2E7D32")
    private val redCol = Color.parseColor("#C62828")

    private fun angleSil(a: Angle): Int = when (a) {
        Angle.FRONT -> R.drawable.ic_hjort_front
        Angle.SKRAA30, Angle.SKRAA60 -> R.drawable.ic_hjort_skraa
        else -> R.drawable.ic_hjort_side
    }

    private fun greyTint(): Int {
        val night = (resources.configuration.uiMode and
            Configuration.UI_MODE_NIGHT_MASK) == Configuration.UI_MODE_NIGHT_YES
        return if (night) Color.parseColor("#3A3A3A") else Color.parseColor("#CFCFCF")
    }

    override fun rebuild() {
        val a = requireActivity()
        val store = Store.get(a)
        content.removeAllViews()

        val evidence = store.currentSeasonSeries().filter { it.countsInEvidence }

        content.addView(Ui.title(a, getString(R.string.tab_innsikt)))
        content.addView(Ui.body(a, getString(R.string.innsikt_recommend)))

        // Jeger-stilling øverst (JJJJ) + (i) høyrejustert
        val topRow = Ui.row(a)
        Position.hoved.forEach { p ->
            topRow.addView(Ui.choiceButton(a, p.label, p == jegerPos, small = true) {
                jegerPos = p; rebuild()
            }.apply { layoutParams = LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply {
                marginEnd = Ui.dp(a, 3) } })
        }
        topRow.addView(ImageButton(a).apply {
            setImageResource(R.drawable.ic_info)
            background = null
            contentDescription = getString(R.string.innsikt_info_title)
            setOnClickListener { infoDialog() }
        })
        content.addView(topRow)

        // Matrise: vilttype-rader til venstre, hold-kolonne til høyre
        val body = LinearLayout(a).apply { orientation = LinearLayout.HORIZONTAL }
        val rows = LinearLayout(a).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }
        species.forEach { sp -> rows.addView(speciesRow(sp, evidence, store)) }
        body.addView(rows)
        body.addView(holdColumn(a))
        content.addView(body)

        // Dyr-vinkling nederst (DDD): tre silhuetter (bredside forhåndsvalgt)
        val bottomRow = Ui.row(a).apply { gravity = Gravity.CENTER }
        listOf(Angle.FRONT to R.drawable.ic_hjort_front,
            Angle.SIDE to R.drawable.ic_hjort_side,
            Angle.SKRAA30 to R.drawable.ic_hjort_skraa).forEach { (ang, res) ->
            bottomRow.addView(ImageView(a).apply {
                setImageResource(res)
                scaleType = ImageView.ScaleType.FIT_CENTER
                val p = Ui.dp(a, 4); setPadding(p, p, p, p)
                alpha = if (dyrAngle == ang) 1f else 0.4f
                layoutParams = LinearLayout.LayoutParams(0, Ui.dp(a, 64), 1f)
                setOnClickListener { dyrAngle = ang; rebuild() }
            })
        }
        content.addView(bottomRow)

        merStatistikkButton(store)
    }

    private fun holdColumn(a: android.app.Activity): LinearLayout {
        val col = LinearLayout(a).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
        }
        holds.forEach { d ->
            col.addView(Ui.choiceButton(a, "$d m", holdM == d, small = true) {
                holdM = d; rebuild()
            }.apply { layoutParams = LinearLayout.LayoutParams(
                Ui.dp(a, 64), ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                bottomMargin = Ui.dp(a, 2) } })
        }
        return col
    }

    private fun speciesRow(sp: Species, evidence: List<SeriesRecord>,
                           store: Store): LinearLayout {
        val a = requireActivity()
        val row = LinearLayout(a).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, Ui.dp(a, 2), 0, Ui.dp(a, 2))
        }

        val posSeries = evidence.filter { it.position == jegerPos }
        val sigma = Stats.sigmaCmAt100(posSeries)
        val n = Stats.shotCount(posSeries)
        val radius = Stats.lethalRadiusCm(sp, dyrAngle)
        val trained = sigma != null && n > 0 && radius != null

        // Silhuett i fast ramme, skalert med hold (25m fyller, 200m halv)
        val scale = (1.10 - (holdM - 25) / 175.0 * 0.60).coerceIn(0.5, 1.10).toFloat()
        val sil = ImageView(a).apply {
            setImageResource(angleSil(dyrAngle))
            scaleType = ImageView.ScaleType.FIT_CENTER
            scaleX = scale; scaleY = scale
            layoutParams = LinearLayout.LayoutParams(
                Ui.dp(a, 96), Ui.dp(a, 72))
            if (!trained) setColorFilter(greyTint())
        }
        row.addView(sil)

        val text = TextView(a).apply {
            textSize = 16f
            layoutParams = LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply {
                marginStart = Ui.dp(a, 8)
            }
        }
        if (!trained) {
            text.text = getString(R.string.innsikt_train_pos)
            text.setTextColor(greyTint())
        } else {
            val p = Stats.pLethal(sigma!!, holdM, radius!!)
            val x = (p * n).roundToInt()
            val goalMet = p >= (1.0 - store.rateLimit)
            text.text = "${sp.label}\n$x av $n\n${(p * 100).roundToInt()} %"
            text.setTextColor(if (goalMet) greenCol else redCol)
        }
        row.addView(text)
        return row
    }

    private fun infoDialog() {
        val a = requireActivity()
        // Fargelegg «Grønn»/«rød» i forklaringsteksten (musingsUI runde 5)
        val raw = getString(R.string.innsikt_info_body)
        val sb = SpannableStringBuilder(raw)
        colorWord(sb, "Grønn", greenCol); colorWord(sb, "grønn", greenCol)
        colorWord(sb, "rød", redCol)
        AlertDialog.Builder(a)
            .setTitle(R.string.innsikt_info_title)
            .setMessage(sb)
            .setPositiveButton(R.string.ok, null).show()
    }

    private fun colorWord(sb: SpannableStringBuilder, word: String, color: Int) {
        var i = sb.indexOf(word)
        while (i >= 0) {
            sb.setSpan(ForegroundColorSpan(color), i, i + word.length,
                Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
            i = sb.indexOf(word, i + word.length)
        }
    }

    private fun merStatistikkButton(store: Store) {
        val a = requireActivity()
        content.addView(MaterialButton(a, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            text = getString(R.string.menu_more_stats)
            layoutParams = Ui.matchWrap(20, a)
            setOnClickListener {
                val evid = store.currentSeasonSeries().filter { it.countsInEvidence }
                val sigma = Stats.sigmaCmAt100(evid)
                val msg = if (sigma == null) getString(R.string.stats_none)
                else "σ (100 m-ekvivalent): %.1f cm\nR95: %.1f cm\nSpredning: %.2f MOA\n\nMålt om siktepunktet (%d skudd)."
                    .format(sigma, Stats.r95Cm(sigma), Stats.moa(sigma), Stats.shotCount(evid))
                AlertDialog.Builder(a).setTitle(R.string.menu_more_stats)
                    .setMessage(msg).setPositiveButton(R.string.ok, null).show()
            }
        })
    }
}
