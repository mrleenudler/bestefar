package no.bestefar.app

import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.view.Gravity
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
import android.content.res.ColorStateList
import com.google.android.material.floatingactionbutton.ExtendedFloatingActionButton

/**
 * Skall (spec §2): seks faner — Våpen, Avstand, Stilling, Innsikt, Jakt,
 * Meny — pluss stor sentrert «Scan serie»-knapp som starter capture-løkka.
 * Økt-flaten er startvisningen; tilbake-knappen går dit før appen lukkes.
 */
class MainActivity : AppCompatActivity() {

    private data class Tab(val iconRes: Int, val labelRes: Int, val make: () -> Fragment)

    private val tabs = listOf(
        Tab(R.drawable.ic_tab_vapen, R.string.tab_vapen) { VapenFragment() },
        Tab(R.drawable.ic_tab_avstand, R.string.tab_avstand) { AvstandFragment() },
        Tab(R.drawable.ic_tab_stilling, R.string.tab_stilling) { StillingFragment() },
        Tab(R.drawable.ic_tab_innsikt, R.string.tab_innsikt) { InnsiktFragment() },
        Tab(R.drawable.ic_tab_jakt, R.string.tab_jakt) { JaktFragment() },
        Tab(R.drawable.ic_tab_meny, R.string.tab_meny) { MenyFragment() },
    )

    private val tabIcons = mutableListOf<ImageView>()
    private val tabLabels = mutableListOf<TextView>()
    private var selected = -1   // -1 = Økt-flaten (hjem)
    private lateinit var store: Store

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = Store.get(this)
        if (!store.onboardingDone) {
            startActivity(Intent(this, OnboardingActivity::class.java))
        }

        val root = FrameLayout(this)
        val column = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }

        val content = FrameLayout(this).apply {
            id = R.id.content_frame
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f)
        }
        column.addView(content)

        val bar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, Ui.dp(this@MainActivity, 64))
            setBackgroundColor(Ui.themeColor(this@MainActivity,
                com.google.android.material.R.attr.colorSurfaceVariant))
        }
        tabs.forEachIndexed { i, tab ->
            val cell = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER
                layoutParams = LinearLayout.LayoutParams(0,
                    ViewGroup.LayoutParams.MATCH_PARENT, 1f)
                setOnClickListener { select(i) }
            }
            val icon = ImageView(this).apply {
                setImageResource(tab.iconRes)
                layoutParams = LinearLayout.LayoutParams(
                    Ui.dp(this@MainActivity, 24), Ui.dp(this@MainActivity, 24))
                contentDescription = getString(tab.labelRes)   // WCAG (spec §9)
            }
            val label = TextView(this).apply {
                text = getString(tab.labelRes)
                textSize = 11f
            }
            cell.addView(icon); cell.addView(label)
            tabIcons.add(icon); tabLabels.add(label)
            bar.addView(cell)
        }
        column.addView(bar)
        root.addView(column, ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT)

        // Stor sentrert scan-knapp (spec §2); kobling til CV-kjernen (spec §11)
        val fab = ExtendedFloatingActionButton(this).apply {
            text = getString(R.string.scan_series)
            setOnClickListener {
                startActivity(Intent(this@MainActivity, CaptureActivity::class.java))
            }
        }
        root.addView(fab, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT,
            Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL
        ).apply { bottomMargin = Ui.dp(this@MainActivity, 84) })

        setContentView(root)
        showHome()

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (selected != -1) showHome() else finish()
            }
        })
    }

    private fun select(i: Int) {
        selected = i
        supportFragmentManager.beginTransaction()
            .replace(R.id.content_frame, tabs[i].make())
            .commit()
        tintTabs()
    }

    private fun showHome() {
        selected = -1
        supportFragmentManager.beginTransaction()
            .replace(R.id.content_frame, OktFragment())
            .commit()
        tintTabs()
    }

    private fun tintTabs() {
        val active = Ui.themeColor(this, com.google.android.material.R.attr.colorPrimary)
        val idle = Color.argb(140, 128, 128, 128)
        tabIcons.forEachIndexed { i, icon ->
            val c = if (i == selected) active else idle
            ImageViewCompat.setImageTintList(icon, ColorStateList.valueOf(c))
            tabLabels[i].setTextColor(c)
        }
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
