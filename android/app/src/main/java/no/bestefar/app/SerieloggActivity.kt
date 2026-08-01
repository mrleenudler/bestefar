package no.bestefar.app

import android.graphics.Color
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter

/**
 * Serielogg (musingsUI): alle scannede serier med dato/tid, avstand,
 * stilling og gjennomsnittspoeng. Klikk åpner serien (Slett serie / OK).
 * Trykk-og-hold gir velg-flere med søppelbøtte øverst til høyre og
 * Avbryt nederst til høyre.
 */
class SerieloggActivity : AppCompatActivity() {

    private lateinit var store: Store
    private lateinit var root: FrameLayout
    private var selectionMode = false
    private val selected = mutableSetOf<String>()
    private var detail: SeriesRecord? = null
    private var seasonOnly = true   // «Denne sesongen» / «Alle» (musingsUI r4)

    // Radene i lista + de to «flytende» knappene holdes som felt, slik at
    // merking KUN oppdaterer det som faktisk endrer seg. Full renderList() ved
    // hver avkryssing scrollet skjermen til toppen (musingsUI runde 10).
    private val rowViews = mutableMapOf<String, TextView>()
    private var trashBtn: ImageButton? = null
    private var cancelBtn: MaterialButton? = null
    private var closeBtn: MaterialButton? = null

    // Karusell over de viste seriene (musingsUI runde 7)
    private var detailList: List<SeriesRecord> = emptyList()
    private var lastShownDate: String? = null
    private var dateAnimator: android.view.ViewPropertyAnimator? = null

    private val fmt = DateTimeFormatter.ofPattern("d.M.yy HH:mm")
    private val dateFmt = ResultActivity.DATE_TIME_FMT

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = Store.get(this)
        root = FrameLayout(this)
        Ui.applyInsets(root)
        setContentView(root)
        renderList()

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                when {
                    detail != null -> { detail = null; renderList() }
                    selectionMode -> exitSelection()
                    else -> finish()
                }
            }
        })
    }

    private fun exitSelection() {
        selectionMode = false
        selected.clear()
        // Ingen ombygging: bare skru av markeringene og bytte knapper tilbake.
        if (rowViews.isEmpty()) renderList() else updateSelectionUi()
    }

    /**
     * Oppdaterer KUN det merkingen påvirker (radbakgrunn + hvilke knapper som
     * vises). Ingen renderList() -> ScrollView beholder posisjonen sin
     * (musingsUI runde 10).
     */
    private fun updateSelectionUi() {
        rowViews.forEach { (id, v) ->
            if (selectionMode && id in selected)
                v.setBackgroundColor(Color.argb(60, 128, 128, 128))
            else v.background = null
        }
        trashBtn?.visibility = if (selectionMode) View.VISIBLE else View.GONE
        cancelBtn?.visibility = if (selectionMode) View.VISIBLE else View.GONE
        closeBtn?.visibility = if (selectionMode) View.GONE else View.VISIBLE
    }

    private fun renderList() {
        root.removeAllViews()
        rowViews.clear()
        val content = Ui.col(this)

        val header = Ui.row(this)
        header.addView(Ui.title(this, getString(R.string.menu_series)).apply {
            layoutParams = LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        })
        // Søppelbøtte øverst i høyre hjørne (musingsUI); ligger alltid i treet
        // og vises kun i merkemodus, så merking ikke krever ombygging.
        trashBtn = ImageButton(this).apply {
            setImageResource(R.drawable.ic_delete)
            background = null
            visibility = if (selectionMode) View.VISIBLE else View.GONE
            contentDescription = getString(R.string.serielogg_delete)
            setOnClickListener { confirmDelete(selected.toSet()) }
        }
        header.addView(trashBtn)
        content.addView(header)

        // Denne sesongen / Alle (musingsUI runde 4)
        val toggle = Ui.row(this)
        toggle.addView(Ui.choiceButton(this, getString(R.string.serier_season), seasonOnly) {
            if (!seasonOnly) { seasonOnly = true; renderList() }
        }.apply { layoutParams = LinearLayout.LayoutParams(0,
            ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply {
            marginEnd = Ui.dp(this@SerieloggActivity, 6) } })
        toggle.addView(Ui.choiceButton(this, getString(R.string.serier_all), !seasonOnly) {
            if (seasonOnly) { seasonOnly = false; renderList() }
        }.apply { layoutParams = LinearLayout.LayoutParams(0,
            ViewGroup.LayoutParams.WRAP_CONTENT, 1f) })
        content.addView(toggle)

        val source = if (seasonOnly) store.currentSeasonSeries() else store.allSeries()
        val all = source.sortedByDescending { it.ts }
        if (all.isEmpty()) {
            content.addView(Ui.hint(this, getString(R.string.serielogg_empty)))
        }
        all.forEachIndexed { index, s ->
            val t = Instant.ofEpochMilli(s.ts).atZone(ZoneId.systemDefault()).format(fmt)
            val avg = if (s.shots.isEmpty()) 0.0 else s.sumDecimal / s.shots.size
            val row = TextView(this).apply {
                text = "$t · ${s.distanceM} m · ${s.position.label} · " +
                    getString(R.string.serielogg_avg, avg)
                textSize = 16f
                setPadding(Ui.dp(this@SerieloggActivity, 8),
                    Ui.dp(this@SerieloggActivity, 14),
                    Ui.dp(this@SerieloggActivity, 8),
                    Ui.dp(this@SerieloggActivity, 14))
                if (selectionMode && s.id in selected) {
                    setBackgroundColor(Color.argb(60, 128, 128, 128))
                }
                setOnClickListener {
                    if (selectionMode) {
                        if (s.id in selected) selected.remove(s.id) else selected.add(s.id)
                        if (selected.isEmpty()) exitSelection() else updateSelectionUi()
                    } else {
                        detailList = all
                        lastShownDate = null
                        detail = s
                        renderDetail(index)
                    }
                }
                setOnLongClickListener {
                    if (!selectionMode) {
                        selectionMode = true
                        selected.add(s.id)
                        updateSelectionUi()
                    }
                    true
                }
            }
            rowViews[s.id] = row
            content.addView(row)
        }

        // Lista skal AVSLUTTE over bunnknappen, ikke scrolle bak den (musingsUI
        // runde 9): scroll-området får bunnmarg så seriene klippes ved knappens
        // overkant (ScrollView klipper egne barn til sine grenser).
        root.addView(Ui.scroll(this, content), FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT).apply {
            bottomMargin = Ui.dp(this@SerieloggActivity, 72)
        })

        // Begge bunnknappene ligger alltid i treet; kun synligheten byttes ved
        // merking (musingsUI runde 10 — unngår ombygging og scroll-hopp).
        val corner = { FrameLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM or Gravity.END).apply {
            bottomMargin = Ui.dp(this@SerieloggActivity, 16)
            rightMargin = Ui.dp(this@SerieloggActivity, 16)
        } }
        // Avbryt nederst i høyre hjørne (musingsUI)
        cancelBtn = MaterialButton(this).apply {
            text = getString(R.string.cancel)
            visibility = if (selectionMode) View.VISIBLE else View.GONE
            setOnClickListener { exitSelection() }
        }
        root.addView(cancelBtn, corner())
        // Lukk nederst i høyre hjørne (musingsUI runde 8)
        closeBtn = MaterialButton(this).apply {
            text = getString(R.string.close)
            visibility = if (selectionMode) View.GONE else View.VISIBLE
            setOnClickListener { finish() }
        }
        root.addView(closeBtn, corner())
    }

    /**
     * Karusell over de viste seriene (musingsUI runde 7): samme rike poeng-
     * visning som resultatkortet, piler i endene (fjernes ytterst) + OK.
     * Dato animeres når den bytter mellom serier.
     */
    private fun renderDetail(index: Int) {
        dateAnimator?.cancel()
        if (detailList.isEmpty() || index !in detailList.indices) { renderList(); return }
        val s = detailList[index]
        detail = s
        root.removeAllViews()
        // Lista er ute av visningstreet -> glem radreferansene (runde 10)
        rowViews.clear(); trashBtn = null; cancelBtn = null; closeBtn = null
        val content = Ui.col(this)

        val zoned = Instant.ofEpochMilli(s.ts).atZone(ZoneId.systemDefault())
        val dateText = zoned.format(dateFmt)
        val dayKey = zoned.toLocalDate().toString()   // animasjon kun ved ny DAG
        val dateView = Ui.title(this, dateText)
        content.addView(dateView)

        val mainRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        mainRow.addView(TargetView(this).apply {
            hits = s.shots
            layoutParams = LinearLayout.LayoutParams(0, Ui.dp(this@SerieloggActivity, 280), 1f)
        })
        val scoreCol = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(Ui.dp(this@SerieloggActivity, 8), 0, 0, 0)
        }
        scoreCol.addView(TextView(this).apply {
            text = getString(R.string.result_points_label)   // «Poeng:»
            // Samme størrelse som poengene, i fet (musingsUI runde 8)
            textSize = 18f
            setTypeface(typeface, android.graphics.Typeface.BOLD)
        })
        s.shots.sortedBy { it.decimal }.forEach { shot ->
            scoreCol.addView(TextView(this).apply {
                text = "%.1f".format(shot.decimal)
                textSize = 18f
            })
        }
        mainRow.addView(scoreCol)
        content.addView(mainRow)

        content.addView(TextView(this).apply {
            text = getString(R.string.result_points_prefix, s.sumDecimal, s.sumInteger)
            textSize = 26f
        })
        val w = store.weapons().firstOrNull { it.id == s.weaponId }
        val modText = when (s.modifier) {
            PosModifier.ANLEGG -> " med anlegg"
            PosModifier.REIM -> " med reim"
            else -> ""
        }
        content.addView(Ui.hint(this,
            "${s.position.label}$modText · ${s.distanceM} m · " +
            (w?.shownName ?: "—") + if (s.corrected) " · korrigert" else ""))

        // Mitt gjennomsnitt for denne stillingen: sesong + totalt, brutt ned kun
        // på stilling (musingsUI runde 8)
        val samePos = store.allSeries().filter { it.position == s.position }
        val seasonNow = Store.seasonKey(s.ts)
        val seasonPos = samePos.filter { Store.seasonKey(it.ts) == seasonNow }
        content.addView(TextView(this).apply {
            text = getString(R.string.result_avg_pos_title)
            textSize = 15f
            setTypeface(typeface, android.graphics.Typeface.BOLD)
            setPadding(0, Ui.dp(this@SerieloggActivity, 8), 0, Ui.dp(this@SerieloggActivity, 2))
        })
        if (seasonPos.isNotEmpty())
            content.addView(Ui.body(this, getString(R.string.result_avg_season,
                seasonPos.map { it.sumDecimal }.average())))
        content.addView(Ui.body(this, getString(R.string.result_avg_total,
            samePos.map { it.sumDecimal }.average())))

        content.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = getString(R.string.serielogg_delete)
            setOnClickListener { confirmDelete(setOf(s.id)) }
        })

        content.addView(android.widget.Space(this), LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, Ui.dp(this, 90)))
        root.addView(Ui.scroll(this, content), ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT)

        // Fast knapperad: piler i endene (fjernes ytterst), OK sentrert
        val bar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
        }
        val left = FrameLayout(this)
        if (index < detailList.size - 1) left.addView(bigArrow("‹") { renderDetail(index + 1) })
        bar.addView(left, LinearLayout.LayoutParams(0,
            ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        bar.addView(MaterialButton(this).apply {
            text = getString(R.string.ok)
            minWidth = Ui.dp(this@SerieloggActivity, 120)
            setOnClickListener { detail = null; renderList() }
        })
        val right = FrameLayout(this)
        if (index > 0) right.addView(bigArrow("›") { renderDetail(index - 1) })
        bar.addView(right, LinearLayout.LayoutParams(0,
            ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        root.addView(bar, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT,
            Gravity.BOTTOM).apply {
            bottomMargin = Ui.dp(this@SerieloggActivity, 24)
            leftMargin = Ui.dp(this@SerieloggActivity, 8)
            rightMargin = Ui.dp(this@SerieloggActivity, 8)
        })

        // Dato-animasjon når datoen bytter (musingsUI runde 7): starter ~3x og
        // ~20 % ned til høyre, zoomer inn til default på 1 s. Ikke-blokkerende;
        // avbrytes hvis pilen trykkes på nytt (dateAnimator.cancel over).
        if (lastShownDate != null && lastShownDate != dayKey) {
            dateView.post {
                dateView.pivotX = 0f; dateView.pivotY = dateView.height.toFloat()
                dateView.scaleX = 3f; dateView.scaleY = 3f
                dateView.translationX = resources.displayMetrics.widthPixels * 0.20f
                dateView.translationY = Ui.dp(this, 40).toFloat()
                dateAnimator = dateView.animate()
                    .scaleX(1f).scaleY(1f).translationX(0f).translationY(0f)
                    .setDuration(1000L)
                    .setInterpolator(android.view.animation.DecelerateInterpolator())
                dateAnimator?.start()
            }
        }
        lastShownDate = dayKey
    }

    /** Store, høye piler — samme stil som ellers der vi blar (runde 7). */
    private fun bigArrow(glyph: String, onClick: () -> Unit) = MaterialButton(this, null,
        com.google.android.material.R.attr.borderlessButtonStyle).apply {
        text = glyph; textSize = 40f; scaleY = 1.7f
        setOnClickListener { onClick() }
    }

    private fun confirmDelete(ids: Set<String>) {
        AlertDialog.Builder(this)
            .setMessage(resources.getQuantityString(
                R.plurals.serielogg_delete_confirm, ids.size, ids.size))
            .setPositiveButton(R.string.serielogg_delete) { _, _ ->
                store.deleteSeries(ids)
                detail = null
                selectionMode = false
                selected.clear()
                renderList()          // lista er endret -> full ombygging
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }
}
