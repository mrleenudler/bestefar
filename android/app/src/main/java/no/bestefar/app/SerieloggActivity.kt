package no.bestefar.app

import android.graphics.Color
import android.os.Bundle
import android.view.Gravity
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

    private val fmt = DateTimeFormatter.ofPattern("d.M.yy HH:mm")

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
        renderList()
    }

    private fun renderList() {
        root.removeAllViews()
        val content = Ui.col(this)

        val header = Ui.row(this)
        header.addView(Ui.title(this, getString(R.string.menu_series)).apply {
            layoutParams = LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        })
        if (selectionMode) {
            // Søppelbøtte øverst i høyre hjørne (musingsUI)
            header.addView(ImageButton(this).apply {
                setImageResource(R.drawable.ic_delete)
                background = null
                contentDescription = getString(R.string.serielogg_delete)
                setOnClickListener { confirmDelete(selected.toSet()) }
            })
        }
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
        all.forEach { s ->
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
                        if (selected.isEmpty()) exitSelection() else renderList()
                    } else {
                        detail = s
                        renderDetail(s)
                    }
                }
                setOnLongClickListener {
                    if (!selectionMode) {
                        selectionMode = true
                        selected.add(s.id)
                        renderList()
                    }
                    true
                }
            }
            content.addView(row)
        }

        root.addView(Ui.scroll(this, content), ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT)

        if (selectionMode) {
            // Avbryt nederst i høyre hjørne (musingsUI)
            root.addView(MaterialButton(this).apply {
                text = getString(R.string.cancel)
                setOnClickListener { exitSelection() }
            }, FrameLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                Gravity.BOTTOM or Gravity.END).apply {
                bottomMargin = Ui.dp(this@SerieloggActivity, 16)
                rightMargin = Ui.dp(this@SerieloggActivity, 16)
            })
        }
    }

    private fun renderDetail(s: SeriesRecord) {
        root.removeAllViews()
        val content = Ui.col(this)
        val t = Instant.ofEpochMilli(s.ts).atZone(ZoneId.systemDefault()).format(fmt)
        content.addView(Ui.title(this, t))

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
        s.shots.sortedBy { it.decimal }.forEach { shot ->
            scoreCol.addView(TextView(this).apply {
                text = "%.1f".format(shot.decimal)
                textSize = 18f
            })
        }
        mainRow.addView(scoreCol)
        content.addView(mainRow)

        content.addView(TextView(this).apply {
            text = "%.1f  (%d)".format(s.sumDecimal, s.sumInteger)
            textSize = 26f
        })
        val w = store.weapons().firstOrNull { it.id == s.weaponId }
        val mod = if (s.modifier != PosModifier.UTEN) " (${s.modifier.label})" else ""
        content.addView(Ui.hint(this,
            "${s.position.label}$mod · ${s.distanceM} m · " +
            (w?.shownName ?: "—") + if (s.corrected) " · korrigert" else ""))

        val btnRow = Ui.row(this)
        btnRow.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = getString(R.string.serielogg_delete)
            setOnClickListener { confirmDelete(setOf(s.id)) }
        })
        btnRow.addView(android.widget.Space(this), LinearLayout.LayoutParams(0, 1, 1f))
        btnRow.addView(MaterialButton(this).apply {
            text = getString(R.string.ok)
            setOnClickListener { detail = null; renderList() }
        })
        content.addView(btnRow, Ui.matchWrap(12, this))

        root.addView(Ui.scroll(this, content), ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT)
    }

    private fun confirmDelete(ids: Set<String>) {
        AlertDialog.Builder(this)
            .setMessage(resources.getQuantityString(
                R.plurals.serielogg_delete_confirm, ids.size, ids.size))
            .setPositiveButton(R.string.serielogg_delete) { _, _ ->
                store.deleteSeries(ids)
                detail = null
                exitSelection()
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }
}
