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
import android.widget.FrameLayout
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
    private fun trainGrey() = if (night()) Color.parseColor("#9A9A9A") else Color.parseColor("#6E6E6E")
    private fun silGrey() = if (night()) Color.parseColor("#8A8A8A") else Color.parseColor("#9E9E9E")

    /**
     * Aktiv silhuett-/piktogram-farge: sort i lys modus, varm lysebrun
     * (colorPrimary = #D8B79B) i mørk (musingsUI runde 9 — brukeren ville ha den
     * varme brune fargen i mørk visning, ikke den nøytrale tekstfargen).
     */
    private fun silColor() = if (night())
        Ui.themeColor(requireContext(), com.google.android.material.R.attr.colorPrimary)
    else Color.BLACK

    /**
     * Egne skaleringer for Innsikt (musingsUI runde 7): FIT_CENTER i cellen
     * normaliserer allerede SVG-ene, så de store enum-skalaene (for
     * stillingsvelgeren etter scan) doblet opp og ga liggende for liten /
     * sittende for stor. Disse verdiene jevner dem ut.
     */
    private fun innsiktScale(p: Position): Float = when (p) {
        Position.LIGGENDE -> 1.0f
        Position.SITTENDE -> 0.75f
        Position.KNESTAENDE -> 0.8f
        Position.STAAENDE -> 1.0f
    }

    /**
     * Viltsilhuett for art + vinkling (musingsUI runde 9/10): Elg, Villsvin og
     * Villrein har egne side/front-silhuetter; øvrige bruker hjort. Villrein er
     * den eneste med EGEN skrå-silhuett; elg/villsvin mangler den, så skrå
     * bruker deres side-silhuett.
     */
    private fun angleSil(sp: Species, a: Angle): Int {
        val front = a == Angle.FRONT
        return when (sp) {
            Species.ELG -> if (front) R.drawable.ic_elg_front else R.drawable.ic_elg_side
            Species.VILLSVIN ->
                if (front) R.drawable.ic_villsvin_front else R.drawable.ic_villsvin_side
            Species.VILLREIN -> when (a) {
                Angle.FRONT -> R.drawable.ic_rein_front
                Angle.SKRAA30, Angle.SKRAA60 -> R.drawable.ic_rein_skraa
                else -> R.drawable.ic_rein_side
            }
            else -> when (a) {
                Angle.FRONT -> R.drawable.ic_hjort_front
                Angle.SKRAA30, Angle.SKRAA60 -> R.drawable.ic_hjort_skraa
                else -> R.drawable.ic_hjort_side
            }
        }
    }

    // ---- Autoskalert rutenett (musingsUI runde 10) ----------------------------
    // Rammen er 7 like høye rader: stillingsraden øverst + 6 rader i kroppen
    // (5 vilttyper + vinkelraden). Hold-kolonnen har nøyaktig 6 knapper, så
    // 200 m havner rett til høyre for vinkel-/vilt-posisjonsvalgene. Radhøyden
    // regnes ut fra skjermhøyden slik at alt får plass på ÉN skjerm.
    private val gap get() = Ui.dp(requireContext(), 2)
    private var unitH = 0          // radavstand (celle + stripe)
    private var cellH = 0          // selve cellehøyden
    private var cellW = 0          // cellebredde = holdknappbredde

    /** Plass tittel/undertekst/knapp/systemlinjer trenger utenom rutenettet. */
    private fun reservedH() = Ui.dp(requireContext(), 250)

    private fun computeMetrics() {
        val a = requireContext()
        val dm = resources.displayMetrics
        val avail = dm.heightPixels - reservedH()
        unitH = (avail / 7).coerceIn(Ui.dp(a, 34), Ui.dp(a, 66))
        cellH = unitH - gap
        // Fire kolonner må få plass i bredden ved siden av kolonnepaddingen
        cellW = minOf(Ui.dp(a, 66), (dm.widthPixels - Ui.dp(a, 40) - 4 * gap) / 4)
    }

    override fun rebuild() {
        val a = requireActivity()
        val store = Store.get(a)
        content.removeAllViews()
        computeMetrics()

        val evidence = store.currentSeasonSeries().filter { it.countsInEvidence }

        // Tittel + (i) høyrejustert, tydelig synlig (musingsUI runde 7)
        val titleRow = Ui.row(a)
        titleRow.addView(Ui.title(a, getString(R.string.tab_innsikt)).apply {
            layoutParams = LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        })
        // (i): UTF-8-glyf i stedet for SVG-ikonet, som var vanskelig å få synlig
        // (musingsUI runde 8). Tekstfarget, tydelig klikkbart.
        titleRow.addView(TextView(a).apply {
            text = "ⓘ"
            textSize = 26f
            setTextColor(txtColor())
            gravity = Gravity.CENTER
            contentDescription = getString(R.string.innsikt_info_title)
            layoutParams = LinearLayout.LayoutParams(Ui.dp(a, 40), Ui.dp(a, 40))
            val pad = android.util.TypedValue.applyDimension(
                android.util.TypedValue.COMPLEX_UNIT_DIP, 4f, resources.displayMetrics).toInt()
            setPadding(pad, pad, pad, pad)
            isClickable = true
            setOnClickListener { infoDialog() }
        })
        content.addView(titleRow)
        content.addView(Ui.body(a, getString(R.string.innsikt_recommend)))

        // Øverste ramme-kant: alle fire stillinger inntil hverandre til venstre,
        // kun en smal stripe imellom (musingsUI runde 9). Stående (siste) står
        // dermed i 4. kolonne — rett over hold-kolonnen.
        val topRow = LinearLayout(a).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, unitH)
        }
        Position.hoved.forEachIndexed { i, p ->
            val last = i == Position.hoved.size - 1
            topRow.addView(iconCell(p.iconRes, innsiktScale(p), p == jegerPos,
                p.label, last = last) { jegerPos = p; rebuild() })
        }
        content.addView(topRow)

        // Kroppen: venstrekolonne (5 vilt-rader + vinkelraden) og hold-kolonnen
        // (6 knapper) — like høye, så 200 m står RETT TIL HØYRE for vinkel-/
        // vilt-posisjonsvalgene (musingsUI runde 10). Venstrekolonnens bredde er
        // låst til de tre første kolonnene, så hold-kolonnen lander under stående.
        val body = LinearLayout(a).apply { orientation = LinearLayout.HORIZONTAL }
        val leftCol = LinearLayout(a).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(3 * cellW + 3 * gap,
                ViewGroup.LayoutParams.WRAP_CONTENT)
        }
        speciesList.forEach { sp -> leftCol.addView(speciesRow(sp, evidence, store)) }

        // Nederste ramme-kant: dyr-vinkling (DDD), venstrejustert under de tre
        // første stillingene. Generisk hjort-silhuett som vinkel-indikator (runde 7).
        val angleRow = LinearLayout(a).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.START
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, unitH)
        }
        listOf(Angle.FRONT to R.drawable.ic_hjort_front,
            Angle.SIDE to R.drawable.ic_hjort_side,
            Angle.SKRAA30 to R.drawable.ic_hjort_skraa).forEach { (ang, res) ->
            angleRow.addView(iconCell(res, 1f, dyrAngle == ang, ang.label, last = false) {
                dyrAngle = ang; rebuild()
            })
        }
        leftCol.addView(angleRow)
        body.addView(leftCol)

        val holdCol = LinearLayout(a).apply {
            orientation = LinearLayout.VERTICAL; gravity = Gravity.CENTER_HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT)
        }
        holds.forEach { d -> holdCol.addView(holdButton(d)) }
        body.addView(holdCol)
        content.addView(body)

        merStatistikkButton(store)
    }

    /** Hold-knapp: alltid tre sifre og «m» synlig, like bred som stående-cellen. */
    private fun holdButton(d: Int): MaterialButton {
        val a = requireActivity()
        val selected = holdM == d
        val b = if (selected) MaterialButton(a)
        else MaterialButton(a, null, com.google.android.material.R.attr.materialButtonOutlinedStyle)
        b.text = "$d m"
        // Skalerer med radhøyden så teksten aldri blir klippet (runde 10)
        b.textSize = (cellH / resources.displayMetrics.density * 0.30f).coerceIn(11f, 15f)
        b.maxLines = 1
        b.isAllCaps = false
        b.cornerRadius = Ui.dp(a, 6)
        b.insetTop = 0; b.insetBottom = 0
        b.minWidth = 0; b.minimumWidth = 0
        b.minHeight = 0; b.minimumHeight = 0
        val ph = Ui.dp(a, 2)
        b.setPadding(ph, 0, ph, 0)
        // Like store som vilt-/stilling-ikoncellene, smal stripe imellom (runde 9)
        b.layoutParams = LinearLayout.LayoutParams(cellW, cellH).apply {
            bottomMargin = gap
        }
        b.setOnClickListener { holdM = d; rebuild() }
        return b
    }

    /**
     * Ramme-innkapslet ikon-knapp. Rammen ligger på en FrameLayout av fast
     * størrelse, og bare det indre bildet skaleres — slik unngås de loddrette
     * «stripene» fra å skalere hele knappen (background) med (musingsUI runde 7).
     * Valgt = sort (lys) / tekstfarge (mørk); uvalgt = grå.
     */
    private fun iconCell(res: Int, scale: Float, selected: Boolean, desc: String,
                         last: Boolean, onClick: () -> Unit): FrameLayout {
        val a = requireActivity()
        val tint = if (selected) silColor() else silGrey()
        val box = FrameLayout(a).apply {
            background = GradientDrawable().apply {
                cornerRadius = Ui.dp(a, 8).toFloat()
                setStroke(Ui.dp(a, if (selected) 2 else 1),
                    if (selected) Ui.themeColor(a,
                        com.google.android.material.R.attr.colorPrimary) else Color.GRAY)
            }
            contentDescription = desc
            layoutParams = LinearLayout.LayoutParams(cellW, cellH).apply {
                if (!last) marginEnd = gap
            }
            setOnClickListener { onClick() }
        }
        val p = Ui.dp(a, 4)
        box.addView(ImageView(a).apply {
            setImageResource(res)
            scaleType = ImageView.ScaleType.FIT_CENTER
            scaleX = scale; scaleY = scale
            setPadding(p, p, p, p)
            ImageViewCompat.setImageTintList(this,
                android.content.res.ColorStateList.valueOf(tint))
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
        })
        return box
    }

    private fun speciesRow(sp: Species, evidence: List<SeriesRecord>,
                           store: Store): LinearLayout {
        val a = requireActivity()
        val row = LinearLayout(a).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, unitH)
        }

        val posSeries = evidence.filter { it.position == jegerPos }
        val sigma = Stats.sigmaCmAt100(posSeries)
        val n = Stats.shotCount(posSeries)
        val radius = Stats.lethalRadiusCm(sp, dyrAngle)
        val trained = sigma != null && n > 0 && radius != null

        // Presentasjonssilhuett: art-spesifikk (musingsUI runde 9), sort i lys /
        // varm brun i mørk, skalert med hold (25 m fyller, 200 m halv). Grå kun
        // når data mangler.
        val scale = (1.10 - (holdM - 25) / 175.0 * 0.60).coerceIn(0.5, 1.10).toFloat()
        val sil = ImageView(a).apply {
            setImageResource(angleSil(sp, dyrAngle))
            scaleType = ImageView.ScaleType.FIT_CENTER
            scaleX = scale; scaleY = scale
            // Silhuettboksen følger radhøyden (autoskalering, runde 10)
            layoutParams = LinearLayout.LayoutParams((cellH * 1.25).toInt(), cellH)
            ImageViewCompat.setImageTintList(this,
                android.content.res.ColorStateList.valueOf(
                    if (trained) silColor() else silGrey()))
        }
        row.addView(sil)

        val text = TextView(a).apply {
            textSize = (cellH / resources.displayMetrics.density * 0.28f)
                .coerceIn(12f, 16f)
            layoutParams = LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply {
                marginStart = Ui.dp(a, 6)
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
            layoutParams = Ui.matchWrap(10, a)   // strammere, alt skal på én skjerm
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
