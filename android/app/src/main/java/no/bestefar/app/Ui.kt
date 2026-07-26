package no.bestefar.app

import android.content.Context
import android.graphics.Color
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

/** Små byggeklosser for programmatisk UI (holder fragmentene kompakte). */
object Ui {

    fun dp(c: Context, v: Int): Int = (v * c.resources.displayMetrics.density).toInt()

    fun col(c: Context, padDp: Int = 16): LinearLayout = LinearLayout(c).apply {
        orientation = LinearLayout.VERTICAL
        val p = dp(c, padDp)
        setPadding(p, p, p, p)
    }

    fun row(c: Context): LinearLayout = LinearLayout(c).apply {
        orientation = LinearLayout.HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
    }

    fun scroll(c: Context, content: View): ScrollView = ScrollView(c).apply {
        isFillViewport = true
        addView(content, ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT)
    }

    fun title(c: Context, s: CharSequence): TextView = TextView(c).apply {
        text = s; textSize = 22f
        setPadding(0, 0, 0, dp(c, 8))
    }

    fun section(c: Context, s: CharSequence): TextView = TextView(c).apply {
        text = s; textSize = 17f
        setTextColor(themeColor(c, com.google.android.material.R.attr.colorPrimary))
        setPadding(0, dp(c, 20), 0, dp(c, 4))
    }

    fun body(c: Context, s: CharSequence): TextView = TextView(c).apply {
        text = s; textSize = 15f
        setPadding(0, dp(c, 4), 0, dp(c, 4))
    }

    fun hint(c: Context, s: CharSequence): TextView = TextView(c).apply {
        text = s; textSize = 13f; alpha = 0.65f
        setPadding(0, dp(c, 4), 0, dp(c, 4))
    }

    fun vspace(c: Context, h: Int): View = View(c).apply {
        layoutParams = LinearLayout.LayoutParams(1, dp(c, h))
    }

    fun themeColor(c: Context, attr: Int): Int {
        val tv = TypedValue()
        return if (c.theme.resolveAttribute(attr, tv, true)) tv.data else Color.BLACK
    }

    /**
     * Edge-to-edge-fix (musingsUI): targetSdk 35 tegner under status- og
     * navigasjonslinjen; alle skjermbilder padder rota med systembar-innsets.
     */
    fun applyInsets(root: View) {
        androidx.core.view.ViewCompat.setOnApplyWindowInsetsListener(root) { v, wi ->
            val i = wi.getInsets(androidx.core.view.WindowInsetsCompat.Type.systemBars())
            v.setPadding(i.left, i.top, i.right, i.bottom)
            wi
        }
    }

    fun matchWrap(topDp: Int = 0, c: Context? = null): LinearLayout.LayoutParams =
        LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT).apply {
            if (c != null) topMargin = dp(c, topDp)
        }

    /**
     * Valg-knapp (musingsUI runde 3): uvalgte = tydelig outlined, valgt =
     * invertert (fylt). `small` gir mindre, rektangulær variant
     * (hjelpemidler-knappene).
     */
    fun choiceButton(c: Context, label: CharSequence, selected: Boolean,
                     small: Boolean = false, onClick: () -> Unit):
        com.google.android.material.button.MaterialButton {
        val b = if (selected) {
            com.google.android.material.button.MaterialButton(c)
        } else {
            com.google.android.material.button.MaterialButton(c, null,
                com.google.android.material.R.attr.materialButtonOutlinedStyle)
        }
        b.text = label
        if (small) {
            b.textSize = 13f
            b.cornerRadius = dp(c, 6)
            b.minHeight = dp(c, 40)
            b.minimumHeight = dp(c, 40)
        }
        b.setOnClickListener { onClick() }
        return b
    }

    /** Fritekstfelt: stor forbokstav per default (musingsUI runde 3). */
    fun capitalize(e: android.widget.EditText) {
        e.inputType = android.text.InputType.TYPE_CLASS_TEXT or
            android.text.InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
    }

    /** Maks lengde på visningsnavn (musingsUI runde 4). */
    const val NAME_MAX = 24

    /**
     * Filter for visningsnavn: begrens lengde og til utskrivbar ASCII, så
     * navn deles trygt mellom brukere (musingsUI runde 4).
     */
    fun nameFilters(): Array<android.text.InputFilter> = arrayOf(
        android.text.InputFilter.LengthFilter(NAME_MAX),
        android.text.InputFilter { src, s, e, _, _, _ ->
            val sb = StringBuilder()
            for (i in s until e) { val c = src[i]; if (c.code in 32..126) sb.append(c) }
            if (sb.length == e - s) null else sb
        })

    /** Tekstboks med synlig ramme (kraftigere visuelt hint enn linje). */
    fun boxed(e: android.widget.EditText, c: Context) {
        e.background = android.graphics.drawable.GradientDrawable().apply {
            setColor(Color.TRANSPARENT)
            setStroke(dp(c, 1), Color.GRAY)
            cornerRadius = dp(c, 6).toFloat()
        }
        val p = dp(c, 8)
        e.setPadding(p, p, p, p)
    }
}
