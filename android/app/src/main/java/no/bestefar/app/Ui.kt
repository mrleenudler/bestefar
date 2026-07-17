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

    fun matchWrap(topDp: Int = 0, c: Context? = null): LinearLayout.LayoutParams =
        LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT).apply {
            if (c != null) topMargin = dp(c, topDp)
        }
}
