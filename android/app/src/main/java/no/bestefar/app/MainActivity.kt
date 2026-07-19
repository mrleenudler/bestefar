package no.bestefar.app

import android.content.res.ColorStateList
import android.content.res.Configuration
import android.graphics.Color
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.widget.ImageViewCompat
import androidx.fragment.app.Fragment
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView

/**
 * Hovedskall (musingsUI): valgknappene øverst — stående: to rader à tre
 * (våpen–avstand–jakt / stilling–innsikt–meny); liggende: én rad. Den store
 * scan-knappen ligger i Økt-flaten (nedre halvdel / full bredde liggende).
 * Intro-skjermene er erstattet av et tutorial-overlegg.
 */
class MainActivity : AppCompatActivity() {

    private data class Tab(val iconRes: Int, val labelRes: Int, val make: () -> Fragment)

    // Rekkefølge (musingsUI): rad 1 våpen-avstand-jakt, rad 2 stilling-innsikt-meny
    private val tabs = listOf(
        Tab(R.drawable.ic_menu_rifle, R.string.tab_vapen) { VapenFragment() },
        Tab(R.drawable.ic_menu_distance, R.string.tab_avstand) { AvstandFragment() },
        Tab(R.drawable.ic_menu_moose, R.string.tab_jakt) { JaktFragment() },
        Tab(R.drawable.ic_menu_position, R.string.tab_stilling) { StillingFragment() },
        Tab(R.drawable.ic_menu_stats, R.string.tab_innsikt) { InnsiktFragment() },
        Tab(R.drawable.ic_tab_meny, R.string.tab_meny) { MenyFragment() },
    )

    private val tabIcons = mutableListOf<ImageView>()
    private val tabLabels = mutableListOf<TextView>()
    private var selected = -1   // -1 = Økt-flaten (hjem)
    private lateinit var store: Store
    private lateinit var root: FrameLayout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = Store.get(this)

        root = FrameLayout(this)
        val column = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }

        val landscape =
            resources.configuration.orientation == Configuration.ORIENTATION_LANDSCAPE

        val bar = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Ui.themeColor(this@MainActivity,
                com.google.android.material.R.attr.colorSurfaceVariant))
        }
        if (landscape) {
            bar.addView(buildRow(tabs.indices.toList()))
        } else {
            bar.addView(buildRow(listOf(0, 1, 2)))
            bar.addView(buildRow(listOf(3, 4, 5)))
        }
        column.addView(bar, ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT)

        val content = FrameLayout(this).apply {
            id = R.id.content_frame
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f)
        }
        column.addView(content)

        root.addView(column, ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT)
        setContentView(root)
        showHome()

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (selected != -1) showHome() else finish()
            }
        })

        if (!store.tutorialSeen) {
            root.post { showTutorial() }
        }
    }

    private fun buildRow(indices: List<Int>): LinearLayout {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, Ui.dp(this@MainActivity, 60))
        }
        indices.forEach { i ->
            val tab = tabs[i]
            val cell = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER
                layoutParams = LinearLayout.LayoutParams(0,
                    ViewGroup.LayoutParams.MATCH_PARENT, 1f)
                setOnClickListener { select(i) }
            }
            val icon = ImageView(this).apply {
                setImageResource(tab.iconRes)
                adjustViewBounds = true
                layoutParams = LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT, Ui.dp(this@MainActivity, 26))
                contentDescription = getString(tab.labelRes)   // WCAG (spec §9)
            }
            val label = TextView(this).apply {
                text = getString(tab.labelRes)
                textSize = 11f
            }
            cell.addView(icon); cell.addView(label)
            tabIcons.add(icon); tabLabels.add(label)
            row.addView(cell)
        }
        return row
    }

    fun select(i: Int) {
        selected = i
        supportFragmentManager.beginTransaction()
            .replace(R.id.content_frame, tabs[i].make())
            .commit()
        tintTabs()
    }

    fun showHome() {
        selected = -1
        supportFragmentManager.beginTransaction()
            .replace(R.id.content_frame, OktFragment())
            .commit()
        tintTabs()
    }

    private fun tintTabs() {
        val active = Ui.themeColor(this, com.google.android.material.R.attr.colorPrimary)
        val idle = Ui.themeColor(this, android.R.attr.textColorPrimary)
        tabIcons.forEachIndexed { i, icon ->
            val c = if (i == selected) active else idle
            ImageViewCompat.setImageTintList(icon, ColorStateList.valueOf(c))
            tabLabels[i].setTextColor(c)
        }
    }

    // ---------- Tutorial (erstatter intro, musingsUI) ----------

    private val tutorialSteps = listOf(
        R.string.tutorial_1_title to R.string.tutorial_1_body,
        R.string.tutorial_2_title to R.string.tutorial_2_body,
        R.string.tutorial_3_title to R.string.tutorial_3_body,
        R.string.tutorial_4_title to R.string.tutorial_4_body,
    )

    fun showTutorial() {
        var idx = 0
        val overlay = FrameLayout(this).apply {
            setBackgroundColor(Color.argb(150, 0, 0, 0))
            isClickable = true   // sluk klikk mot UI-et bak
        }
        val card = MaterialCardView(this).apply {
            radius = Ui.dp(this@MainActivity, 16).toFloat()
        }
        val inner = Ui.col(this, 20)
        val title = TextView(this).apply { textSize = 20f }
        val body = TextView(this).apply { textSize = 15f }
        val next = MaterialButton(this)
        fun renderStep() {
            val (t, b) = tutorialSteps[idx]
            title.setText(t); body.setText(b)
            next.text = getString(
                if (idx == tutorialSteps.size - 1) R.string.tutorial_done
                else R.string.tutorial_next)
        }
        next.setOnClickListener {
            if (idx == tutorialSteps.size - 1) {
                store.tutorialSeen = true
                root.removeView(overlay)
            } else {
                idx++; renderStep()
            }
        }
        inner.addView(title); inner.addView(body)
        inner.addView(next, Ui.matchWrap(12, this))
        card.addView(inner)
        overlay.addView(card, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT,
            Gravity.CENTER
        ).apply {
            leftMargin = Ui.dp(this@MainActivity, 24)
            rightMargin = Ui.dp(this@MainActivity, 24)
        })
        renderStep()
        root.addView(overlay, ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT)
    }

    override fun onResume() {
        super.onResume()
        maybeAskFollowUp()
    }

    /**
     * Totrinns utfall (spec §4): ett stille spørsmål om ettersøk ved neste
     * åpning — nøytralt, aldri gjentatt purring (spørres kun én gang).
     */
    private fun maybeAskFollowUp() {
        val rec = store.allHunts().firstOrNull {
            it.outcome == Outcome.SKADE && it.followUp == null && !it.followUpAsked &&
                System.currentTimeMillis() - it.ts > 2 * 60 * 60 * 1000
        } ?: return
        rec.followUpAsked = true
        store.updateHunt(rec)
        val options = FollowUp.entries.map { it.label } +
            getString(R.string.hunt_followup_later)
        AlertDialog.Builder(this)
            .setTitle(R.string.hunt_followup_title)
            .setMessage(R.string.hunt_followup_body)
            .setItems(options.toTypedArray()) { _, i ->
                if (i < FollowUp.entries.size) {
                    rec.followUp = FollowUp.entries[i]
                    store.updateHunt(rec)
                }
            }
            .setCancelable(true)
            .show()
    }
}
