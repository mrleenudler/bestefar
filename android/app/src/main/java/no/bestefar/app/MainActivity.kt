package no.bestefar.app

import android.content.res.ColorStateList
import android.content.res.Configuration
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.Space
import android.widget.TextView
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.widget.ImageViewCompat
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView

/**
 * Hovedskall (musingsUI runde 4): kun tre ikonknapper øverst — Avstand,
 * Innsikt, Meny. Våpen og Jakt ligger nå i menyen; Stilling promptes bare
 * etter hvert scan. Avstand og Meny åpner som dropdown-paneler; Innsikt er
 * fullskjerm. Oppstartsmelding (to vinduer) vises første gang / ved endring.
 */
class MainActivity : AppCompatActivity() {

    companion object {
        const val TAB_AVSTAND = 0
        const val TAB_INNSIKT = 1
        const val TAB_MENY = 2
        /** Bump ved endring av oppstartsmeldingen (vises da på nytt). */
        const val STARTUP_MSG_VERSION = 1
    }

    private val tabIcons = listOf(
        R.drawable.ic_menu_distance, R.drawable.ic_menu_stats, R.drawable.ic_tab_meny)
    private val tabLabels = listOf(
        R.string.tab_avstand, R.string.tab_innsikt, R.string.tab_meny)

    private val boxDrawables = mutableMapOf<Int, GradientDrawable>()
    private var openPanel = -1
    private var innsiktOpen = false
    var reopenMenyOnResume = false
    lateinit var store: Store
    private lateinit var root: FrameLayout
    private lateinit var stack: FrameLayout
    private lateinit var scrim: View
    private lateinit var dropdownWrap: FrameLayout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = Store.get(this)

        root = FrameLayout(this)
        Ui.applyInsets(root)
        val column = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }

        column.addView(buildRow(listOf(TAB_AVSTAND, TAB_INNSIKT, TAB_MENY)),
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)

        stack = FrameLayout(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f)
        }
        val content = FrameLayout(this).apply { id = R.id.content_frame }
        stack.addView(content, ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT)
        scrim = View(this).apply {
            visibility = View.GONE
            setOnClickListener { closeDropdown() }
        }
        stack.addView(scrim, ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT)
        dropdownWrap = FrameLayout(this).apply { visibility = View.GONE }
        stack.addView(dropdownWrap, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT,
            Gravity.TOP))
        column.addView(stack)

        root.addView(column, ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT)
        setContentView(root)
        showHome()

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                when {
                    openPanel != -1 -> closeDropdown()
                    innsiktOpen -> showHome()
                    else -> finish()
                }
            }
        })

        maybeStartupMessage()
    }

    private fun buildRow(indices: List<Int>): LinearLayout {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, Ui.dp(this@MainActivity, 56))
        }
        val landscape =
            resources.configuration.orientation == Configuration.ORIENTATION_LANDSCAPE
        val btnWeight = if (landscape) 0.6f else 0.8f
        val sideWeight = (1f - btnWeight) / 2f
        indices.forEach { i ->
            val cell = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                layoutParams = LinearLayout.LayoutParams(0,
                    ViewGroup.LayoutParams.MATCH_PARENT, 1f)
            }
            val box = GradientDrawable().apply {
                setColor(Color.BLACK)
                setStroke(Ui.dp(this@MainActivity, 1), Color.GRAY)
                cornerRadius = Ui.dp(this@MainActivity, 10).toFloat()
            }
            boxDrawables[i] = box
            // Meny-ikonet er 10 % mindre (musingsUI runde 4) -> mer innramming
            val pad = Ui.dp(this@MainActivity, if (i == TAB_MENY) 9 else 5)
            val icon = ImageView(this).apply {
                setImageResource(tabIcons[i])
                background = box
                scaleType = ImageView.ScaleType.FIT_CENTER
                setPadding(pad, pad, pad, pad)
                ImageViewCompat.setImageTintList(this,
                    ColorStateList.valueOf(Color.WHITE))
                contentDescription = getString(tabLabels[i])
                setOnClickListener { select(i) }
            }
            cell.addView(Space(this), LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.MATCH_PARENT, sideWeight))
            cell.addView(icon, LinearLayout.LayoutParams(0,
                Ui.dp(this@MainActivity, 44), btnWeight))
            cell.addView(Space(this), LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.MATCH_PARENT, sideWeight))
            row.addView(cell)
        }
        return row
    }

    fun select(i: Int) {
        if (i == TAB_INNSIKT) {
            closeDropdown()
            innsiktOpen = true
            supportFragmentManager.beginTransaction()
                .replace(R.id.content_frame, InnsiktFragment())
                .commit()
            tintTabs()
            return
        }
        if (openPanel == i) { closeDropdown(); return }
        openDropdown(i)
    }

    private fun openDropdown(i: Int) {
        if (innsiktOpen) {
            innsiktOpen = false
            supportFragmentManager.beginTransaction()
                .replace(R.id.content_frame, OktFragment())
                .commit()
        }
        dropdownWrap.removeAllViews()
        val panel = Panels.build(i, this) { openDropdown(i) }
        val scroller = android.widget.ScrollView(this).apply {
            addView(panel, ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT)
        }
        val card = MaterialCardView(this).apply {
            radius = Ui.dp(this@MainActivity, 12).toFloat()
            cardElevation = Ui.dp(this@MainActivity, 8).toFloat()
            addView(scroller)
        }
        // Meny inntil høyre; avstand full bredde (musingsUI)
        val lp = if (i == TAB_MENY) FrameLayout.LayoutParams(
            (resources.displayMetrics.widthPixels * 0.62).toInt(),
            ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.END)
        else FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.TOP)
        lp.setMargins(Ui.dp(this, 8), Ui.dp(this, 4), Ui.dp(this, 8), 0)
        dropdownWrap.addView(card, lp)
        val maxH = stack.height - Ui.dp(this, 16)
        if (maxH > 0) card.post {
            if (card.height > maxH) { lp.height = maxH; card.layoutParams = lp }
        }
        dropdownWrap.visibility = View.VISIBLE
        // Kun avstandsmenyen lukkes ved klikk utenfor (musingsUI)
        scrim.visibility = if (i == TAB_AVSTAND) View.VISIBLE else View.GONE
        openPanel = i
        tintTabs()
    }

    fun closeDropdown() {
        dropdownWrap.removeAllViews()
        dropdownWrap.visibility = View.GONE
        scrim.visibility = View.GONE
        openPanel = -1
        tintTabs()
    }

    /** Lukk med 0,5 s forsinkelse (visuell tilbakemelding, musingsUI). */
    fun closeDropdownDelayed() {
        dropdownWrap.postDelayed({ closeDropdown() }, 500)
    }

    fun showHome() {
        closeDropdown()
        innsiktOpen = false
        supportFragmentManager.beginTransaction()
            .replace(R.id.content_frame, OktFragment())
            .commit()
        tintTabs()
    }

    private fun tintTabs() {
        val active = Ui.themeColor(this, com.google.android.material.R.attr.colorPrimary)
        boxDrawables.forEach { (i, box) ->
            val sel = if (innsiktOpen && openPanel == -1) TAB_INNSIKT else openPanel
            box.setStroke(Ui.dp(this, if (i == sel) 2 else 1),
                if (i == sel) active else Color.GRAY)
        }
    }

    // ---------- Oppstartsmelding (musingsUI runde 4) ----------

    private fun maybeStartupMessage() {
        if (store.startupMsgSeenVersion >= STARTUP_MSG_VERSION) {
            if (!store.tutorialSeen) root.post { showTutorial() }
            return
        }
        // Vindu 1: velkomst / prøveversjon (to avsnitt)
        AlertDialog.Builder(this)
            .setTitle(R.string.startup_title)
            .setMessage(R.string.startup_info)
            .setPositiveButton(R.string.ok) { _, _ -> startupWindow2() }
            .setCancelable(false)
            .show()
    }

    private fun startupWindow2() {
        // Vindu 2: deling av bilder der appen gjør en dårlig jobb
        AlertDialog.Builder(this)
            .setMessage(R.string.startup_donate)
            .setPositiveButton(R.string.donate_accept) { _, _ ->
                store.shareDevImages = "ja"; finishStartup()
            }
            .setNegativeButton(R.string.no_thanks) { _, _ ->
                store.shareDevImages = "nei"; finishStartup()
            }
            .setCancelable(false)
            .show()
    }

    private fun finishStartup() {
        store.startupMsgSeenVersion = STARTUP_MSG_VERSION
        if (!store.tutorialSeen) showTutorial()
    }

    // ---------- Tutorial / «Hvordan bruke appen» ----------

    private val tutorialSteps = listOf(
        R.string.tutorial_1_title to R.string.tutorial_1_body,
        R.string.tutorial_2_title to R.string.tutorial_2_body,
        R.string.tutorial_3_title to R.string.tutorial_3_body,
        R.string.tutorial_4_title to R.string.tutorial_4_body,
    )

    fun showTutorial() {
        var idx = 0
        val overlay = FrameLayout(this).apply {
            setBackgroundColor(Color.argb(150, 0, 0, 0)); isClickable = true
        }
        fun dismiss() { store.tutorialSeen = true; root.removeView(overlay) }
        val card = MaterialCardView(this).apply {
            radius = Ui.dp(this@MainActivity, 16).toFloat()
        }
        val inner = Ui.col(this, 20)
        val title = TextView(this).apply { textSize = 20f }
        val body = TextView(this).apply { textSize = 15f }
        val next = MaterialButton(this)
        val skip = MaterialButton(this, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = getString(R.string.tutorial_skip)
            setOnClickListener { dismiss() }
        }
        fun renderStep() {
            val (t, b) = tutorialSteps[idx]
            title.setText(t); body.setText(b)
            next.text = getString(if (idx == tutorialSteps.size - 1)
                R.string.tutorial_done else R.string.tutorial_next)
            skip.visibility = if (idx == tutorialSteps.size - 1) View.GONE else View.VISIBLE
        }
        next.setOnClickListener {
            if (idx == tutorialSteps.size - 1) dismiss() else { idx++; renderStep() }
        }
        inner.addView(title); inner.addView(body)
        val btnRow = Ui.row(this)
        btnRow.addView(skip)
        btnRow.addView(Space(this), LinearLayout.LayoutParams(0, 1, 1f))
        btnRow.addView(next)
        inner.addView(btnRow, Ui.matchWrap(12, this))
        card.addView(inner)
        overlay.addView(card, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT,
            Gravity.CENTER).apply {
            leftMargin = Ui.dp(this@MainActivity, 24); rightMargin = Ui.dp(this@MainActivity, 24)
        })
        renderStep()
        root.addView(overlay, ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT)
    }

    override fun onResume() {
        super.onResume()
        if (reopenMenyOnResume) { reopenMenyOnResume = false; openDropdown(TAB_MENY) }
        maybeAskFollowUp()
    }

    override fun onStop() {
        super.onStop()
        if (openPanel != -1 && !reopenMenyOnResume) closeDropdown()
    }

    /** Totrinns utfall (spec §4): ett stille ettersøks-spørsmål, aldri purret. */
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
                    rec.followUp = FollowUp.entries[i]; store.updateHunt(rec)
                }
            }
            .setCancelable(true)
            .show()
    }
}
