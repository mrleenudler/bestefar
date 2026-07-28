package no.bestefar.app

import android.content.res.Configuration
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.text.style.StyleSpan
import android.view.Gravity
import android.view.ViewGroup
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.core.widget.ImageViewCompat
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
 * Innsikt (musingsUI runde 6): matrise med fem vilttyper rammet av jeger-stilling
 * (ikoner, øverst), dyr-vinkling (silhuetter, nederst) og skuddhold (kolonne
 * t.h.). Jeger-/vilt-/holdknapper er like store. Frekvenstekst grønn = jaktmål
 * nådd, rød ellers; grå «øv på stillingen» der data mangler. (i) på tittellinjen.
 */
class InnsiktFragment : RebuildFragment() {

    private var jegerPos = Position.SITTENDE
    private var dyrAngle = Angle.SIDE
    private var holdM = 100
    private val holds = listOf(25, 50, 75, 100, 150, 200)
    private val speciesList = listOf(Species.ELG, Species.HJORT, Species.VILLREIN,
        Species.RAADYR, Species.VILLSVIN)

    private val greenCol = Color.parseColor("#2E7D32")
    private val redCol = Color.parseColor("#C62828")

    private fun night() = (resources.configuration.uiMode and
        Configuration.UI_MODE_NIGHT_MASK) == Configuration.UI_MODE_NIGHT_YES

    private fun txtColor() = Ui.themeColor(requireContext(), android.R.attr.textColorPrimary)
    private fun muted(c: Int) = Color.argb(90, Color.red(c), Color.green(c), Color.blue(c))
    private fun trainGrey() = if (night()) Color.parseColor("#9A9A9A") else Color.parseColor("#6E6E6E")
    private fun silGrey() = if (night()) Color.parseColor("#555555") else Color.parseColor("#CFCFCF")

    private fun angleSil(a: Angle): Int = when (a) {
        Angle.FRONT -> R.drawable.ic_hjort_front
        Angle.SKRAA30, Angle.SKRAA60 -> R.drawable.ic_hjort_skraa
        else -> R.drawable.ic_hjort_side
    }

    private val cellW get() = Ui.dp(requireContext(), 64)
    private val cellH get() = Ui.dp(requireContext(), 52)

    override fun rebuild() {
        val a = requireActivity()
        val store = Store.get(a)
        content.removeAllViews()

        val evidence = store.currentSeasonSeries().filter { it.countsInEvidence }

        // Tittel + (i) høyrejustert (musingsUI runde 6)
        val titleRow = Ui.row(a)
        titleRow.addView(Ui.title(a, getString(R.string.tab_innsikt)).apply {
            layoutParams = LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        })
        titleRow.addView(ImageButton(a).apply {
            setImageResource(R.drawable.ic_info)
            background = null
            ImageViewCompat.setImageTintList(this,
                android.content.res.ColorStateList.valueOf(txtColor()))
            contentDescription = getString(R.string.innsikt_info_title)
            setOnClickListener { infoDialog() }
        })
        content.addView(titleRow)
        content.addView(Ui.body(a, getString(R.string.innsikt_recommend)))

        // Jeger-stilling som ikoner (JJJJ), sentrert
        val topRow = Ui.row(a).apply { gravity = Gravity.CENTER }
        Position.hoved.forEach { p ->
            topRow.addView(iconCell(p.iconRes, p.iconScale, p == jegerPos,
                p.label) { jegerPos = p; rebuild() })
        }
        content.addView(topRow)

        // Matrise: vilttype-rader (venstre) + hold-kolonne (høyre)
        val body = LinearLayout(a).apply { orientation = LinearLayout.HORIZONTAL }
        val rows = LinearLayout(a).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }
        speciesList.forEach { sp -> rows.addView(speciesRow(sp, evidence, store)) }
        body.addView(rows)

        val holdCol = LinearLayout(a).apply {
            orientation = LinearLayout.VERTICAL; gravity = Gravity.CENTER_HORIZONTAL
        }
        holds.forEach { d ->
            holdCol.addView(Ui.choiceButton(a, "$d m", holdM == d, small = true) {
                holdM = d; rebuild()
            }.apply {
                maxLines = 1
                layoutParams = LinearLayout.LayoutParams(cellW, cellH).apply {
                    bottomMargin = Ui.dp(a, 2)
                }
            })
        }
        body.addView(holdCol)
        content.addView(body)

        // Dyr-vinkling som silhuetter (DDD), sentrert
        val bottomRow = Ui.row(a).apply { gravity = Gravity.CENTER }
        listOf(Angle.FRONT to R.drawable.ic_hjort_front,
            Angle.SIDE to R.drawable.ic_hjort_side,
            Angle.SKRAA30 to R.drawable.ic_hjort_skraa).forEach { (ang, res) ->
            bottomRow.addView(iconCell(res, 1f, dyrAngle == ang, ang.label) {
                dyrAngle = ang; rebuild()
            })
        }
        content.addView(bottomRow)

        merStatistikkButton(store)
    }

    /** Ramme-innkapslet ikon-knapp med tint og valgt/uvalgt-tilstand. */
    private fun iconCell(res: Int, scale: Float, selected: Boolean, desc: String,
                         onClick: () -> Unit): ImageButton {
        val a = requireActivity()
        val tint = if (selected) txtColor() else muted(txtColor())
        return ImageButton(a).apply {
            setImageResource(res)
            scaleType = ImageView.ScaleType.FIT_CENTER
            scaleX = scale; scaleY = scale
            ImageViewCompat.setImageTintList(this,
                android.content.res.ColorStateList.valueOf(tint))
            background = GradientDrawable().apply {
                cornerRadius = Ui.dp(a, 8).toFloat()
                setStroke(Ui.dp(a, if (selected) 2 else 1),
                    if (selected) Ui.themeColor(a,
                        com.google.android.material.R.attr.colorPrimary) else Color.GRAY)
            }
            contentDescription = desc
            val p = Ui.dp(a, 4); setPadding(p, p, p, p)
            layoutParams = LinearLayout.LayoutParams(cellW, cellH).apply {
                marginEnd = Ui.dp(a, 4)
            }
            setOnClickListener { onClick() }
        }
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

        // Fast ramme, silhuett skalert med hold (25 m fyller, 200 m halv)
        val scale = (1.10 - (holdM - 25) / 175.0 * 0.60).coerceIn(0.5, 1.10).toFloat()
        val sil = ImageView(a).apply {
            setImageResource(angleSil(dyrAngle))
            scaleType = ImageView.ScaleType.FIT_CENTER
            scaleX = scale; scaleY = scale
            layoutParams = LinearLayout.LayoutParams(Ui.dp(a, 96), Ui.dp(a, 72))
            ImageViewCompat.setImageTintList(this,
                android.content.res.ColorStateList.valueOf(
                    if (trained) txtColor() else silGrey()))
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
            text.setTextColor(trainGrey())
        } else {
            val p = Stats.pLethal(sigma!!, holdM, radius!!)
            val x = (p * n).roundToInt()
            val goalMet = p >= (1.0 - store.rateLimit)
            val col = if (goalMet) greenCol else redCol
            // Viltnavn i default farge; andel + prosent i fargekode (runde 6)
            val sb = SpannableStringBuilder()
            sb.append(sp.label + "\n")
            val start = sb.length
            sb.append("$x av $n\n${(p * 100).roundToInt()} %")
            sb.setSpan(ForegroundColorSpan(col), start, sb.length,
                Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
            text.setTextColor(txtColor())
            text.text = sb
        }
        row.addView(text)
        return row
    }

    private fun infoDialog() {
        val a = requireActivity()
        val raw = getString(R.string.innsikt_info_body)
        val sb = SpannableStringBuilder(raw)
        styleWord(sb, "Grønn", greenCol); styleWord(sb, "grønn", greenCol)
        styleWord(sb, "rød", redCol)
        AlertDialog.Builder(a)
            .setTitle(R.string.innsikt_info_title)
            .setMessage(sb)
            .setPositiveButton(R.string.ok, null).show()
    }

    /** Fargelegg OG uthev (fet) et ord i teksten (musingsUI runde 6). */
    private fun styleWord(sb: SpannableStringBuilder, word: String, color: Int) {
        var i = sb.indexOf(word)
        while (i >= 0) {
            sb.setSpan(ForegroundColorSpan(color), i, i + word.length,
                Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
            sb.setSpan(StyleSpan(android.graphics.Typeface.BOLD), i, i + word.length,
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
