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
 * Hovedskall (musingsUI-runde 2): ikonknapper i hvitt på sort med grå ramme,
 * uten tekst; stående 2×3 (våpen-avstand-jakt / stilling-innsikt-meny),
 * liggende én rad i rekkefølgen våpen-avstand-stilling-innsikt-jakt-meny.
 * Våpen/avstand/stilling/jakt/meny åpner som dropdown-paneler (trykk igjen
 * lukker); Innsikt er fullskjerm. Tutorial-overlegg med velkomst og skip.
 */
class MainActivity : AppCompatActivity() {

    // Indeks = fane-id brukt av select(): 0 våpen, 1 avstand, 2 jakt,
    // 3 stilling, 4 innsikt, 5 meny
    private val tabIcons = listOf(
        R.drawable.ic_menu_rifle, R.drawable.ic_menu_distance,
        R.drawable.ic_menu_moose, R.drawable.ic_menu_position,
        R.drawable.ic_menu_stats, R.drawable.ic_tab_meny,
    )
    private val tabLabels = listOf(
        R.string.tab_vapen, R.string.tab_avstand, R.string.tab_jakt,
        R.string.tab_stilling, R.string.tab_innsikt, R.string.tab_meny,
    )

    private val boxDrawables = mutableMapOf<Int, GradientDrawable>()
    private var openPanel = -1      // -1 = ingen dropdown åpen
    private var innsiktOpen = false
    lateinit var store: Store
    private lateinit var root: FrameLayout
    private lateinit var dropdownWrap: FrameLayout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = Store.get(this)

        root = FrameLayout(this)
        Ui.applyInsets(root)
        val column = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }

        val landscape =
            resources.configuration.orientation == Configuration.ORIENTATION_LANDSCAPE

        val bar = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        if (landscape) {
            // Jakt nest lengst til høyre (musingsUI)
            bar.addView(buildRow(listOf(0, 1, 3, 4, 2, 5)))
        } else {
            bar.addView(buildRow(listOf(0, 1, 2)))
            bar.addView(buildRow(listOf(3, 4, 5)))
        }
        column.addView(bar, ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT)

        // Innhold + dropdown-lag under knappraden
        val stack = FrameLayout(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f)
        }
        val content = FrameLayout(this).apply { id = R.id.content_frame }
        stack.addView(content, ViewGroup.LayoutParams.MATCH_PARENT,
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

        if (!store.tutorialSeen) {
            root.post { showTutorial() }
        }
    }

    private fun buildRow(indices: List<Int>): LinearLayout {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, Ui.dp(this@MainActivity, 56))
        }
        indices.forEach { i ->
            // Celle med 10 % luft på hver side -> knapp på 80 % av bredden
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
            val icon = ImageView(this).apply {
                setImageResource(tabIcons[i])
                background = box
                scaleType = ImageView.ScaleType.FIT_CENTER
                val p = Ui.dp(this@MainActivity, 8)
                setPadding(p, p, p, p)
                ImageViewCompat.setImageTintList(this,
                    ColorStateList.valueOf(Color.WHITE))
                contentDescription = getString(tabLabels[i])   // WCAG (spec §9)
                setOnClickListener { select(i) }
            }
            cell.addView(Space(this), LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.MATCH_PARENT, 0.1f))
            cell.addView(icon, LinearLayout.LayoutParams(0,
                Ui.dp(this@MainActivity, 44), 0.8f))
            cell.addView(Space(this), LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.MATCH_PARENT, 0.1f))
            row.addView(cell)
        }
        return row
    }

    /** Fane-trykk: Innsikt = fullskjerm; andre = dropdown (trykk igjen lukker). */
    fun select(i: Int) {
        if (i == 4) {
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
        dropdownWrap.removeAllViews()
        val panel = Panels.build(i, this) { openDropdown(i) }
        val card = MaterialCardView(this).apply {
            radius = Ui.dp(this@MainActivity, 12).toFloat()
            cardElevation = Ui.dp(this@MainActivity, 8).toFloat()
            addView(panel)
        }
        // Stilling trekkes til venstre; meny inntil høyre side (musingsUI)
        val lp = when (i) {
            3 -> FrameLayout.LayoutParams(
                (resources.displayMetrics.widthPixels * 0.80).toInt(),
                ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.START)
            5 -> FrameLayout.LayoutParams(
                (resources.displayMetrics.widthPixels * 0.60).toInt(),
                ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.END)
            else -> FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.TOP)
        }
        lp.setMargins(Ui.dp(this, 8), Ui.dp(this, 4), Ui.dp(this, 8), 0)
        dropdownWrap.addView(card, lp)
        dropdownWrap.visibility = View.VISIBLE
        openPanel = i
        tintTabs()
    }

    fun closeDropdown() {
        dropdownWrap.removeAllViews()
        dropdownWrap.visibility = View.GONE
        openPanel = -1
        tintTabs()
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
            val selectedTab = if (innsiktOpen && openPanel == -1) 4 else openPanel
            if (i == selectedTab) {
                box.setStroke(Ui.dp(this, 2), active)
            } else {
                box.setStroke(Ui.dp(this, 1), Color.GRAY)
            }
        }
    }

    // ---------- Tutorial (velkomst + skip, musingsUI-runde 2) ----------

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
        fun dismiss() {
            store.tutorialSeen = true
            root.removeView(overlay)
        }
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
            next.text = getString(
                if (idx == tutorialSteps.size - 1) R.string.tutorial_done
                else R.string.tutorial_next)
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
