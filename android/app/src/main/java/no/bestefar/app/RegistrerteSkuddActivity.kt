package no.bestefar.app

import android.graphics.Color
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Locale

/**
 * «Se registrerte skudd» (musingsUI runde 4/5): liste med vilt-ikon, stedsnavn
 * og dato. Klikk-og-hold merker flere for sletting (Slett øverst t.h., Avbryt
 * nederst t.v.). Klikk åpner detalj med store piler, Rediger øverst t.h. og en
 * fast plassert OK-knapp som ikke flytter seg.
 */
class RegistrerteSkuddActivity : AppCompatActivity() {

    private lateinit var root: FrameLayout
    private lateinit var records: MutableList<HuntRecord>
    private val dateFmt = DateTimeFormatter.ofPattern("d. MMMM yyyy", Locale("no"))
    private var selectionMode = false
    private val selected = mutableSetOf<String>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        root = FrameLayout(this)
        Ui.applyInsets(root)
        setContentView(root)
        reload()
        renderList()
    }

    private fun reload() {
        records = Store.get(this).allHunts().sortedByDescending { it.ts }.toMutableList()
    }

    private fun dateOf(r: HuntRecord) =
        Instant.ofEpochMilli(r.ts).atZone(ZoneId.systemDefault()).toLocalDate().format(dateFmt)

    private fun speciesLabel(r: HuntRecord) =
        if (r.species == Species.ANNET && r.speciesOther.isNotBlank()) r.speciesOther
        else r.species.label

    // ---------- Liste ----------

    private fun renderList() {
        root.removeAllViews()
        val content = Ui.col(this)

        val header = Ui.row(this)
        header.addView(Ui.title(this, getString(R.string.hunt_view_registered)).apply {
            layoutParams = LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        })
        if (selectionMode) {
            header.addView(ImageButton(this).apply {
                setImageResource(R.drawable.ic_delete)
                background = null
                contentDescription = getString(R.string.serielogg_delete)
                setOnClickListener { confirmDelete(selected.toSet()) }
            })
        }
        content.addView(header)

        if (records.isEmpty()) content.addView(Ui.hint(this, getString(R.string.hunt_none)))
        records.forEachIndexed { i, r ->
            val row = Ui.row(this).apply {
                setPadding(Ui.dp(this@RegistrerteSkuddActivity, 4),
                    Ui.dp(this@RegistrerteSkuddActivity, 8),
                    Ui.dp(this@RegistrerteSkuddActivity, 4),
                    Ui.dp(this@RegistrerteSkuddActivity, 8))
                if (selectionMode && r.id in selected)
                    setBackgroundColor(Color.argb(60, 128, 128, 128))
                setOnClickListener {
                    if (selectionMode) {
                        if (r.id in selected) selected.remove(r.id) else selected.add(r.id)
                        if (selected.isEmpty()) exitSelection() else renderList()
                    } else renderDetail(i)
                }
                setOnLongClickListener {
                    if (!selectionMode) { selectionMode = true; selected.add(r.id); renderList() }
                    true
                }
            }
            row.addView(ImageView(this).apply {
                setImageResource(R.drawable.ic_hjort_side)
                layoutParams = LinearLayout.LayoutParams(
                    Ui.dp(this@RegistrerteSkuddActivity, 36),
                    Ui.dp(this@RegistrerteSkuddActivity, 36))
                contentDescription = speciesLabel(r)
            })
            row.addView(TextView(this).apply {
                text = "  ${speciesLabel(r)} · ${r.placeName.ifBlank { "—" }} · ${dateOf(r)}"
                textSize = 16f
            })
            content.addView(row)
        }

        root.addView(Ui.scroll(this, content), ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT)

        if (selectionMode) {
            root.addView(MaterialButton(this).apply {
                text = getString(R.string.cancel)
                setOnClickListener { exitSelection() }
            }, FrameLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM or Gravity.START).apply {
                bottomMargin = Ui.dp(this@RegistrerteSkuddActivity, 16)
                leftMargin = Ui.dp(this@RegistrerteSkuddActivity, 16)
            })
        }
    }

    private fun exitSelection() { selectionMode = false; selected.clear(); renderList() }

    private fun confirmDelete(ids: Set<String>) {
        AlertDialog.Builder(this)
            .setMessage(resources.getQuantityString(
                R.plurals.serielogg_delete_confirm, ids.size, ids.size))
            .setPositiveButton(R.string.serielogg_delete) { _, _ ->
                Store.get(this).deleteHunts(ids); reload(); exitSelection()
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    // ---------- Detalj ----------

    private fun renderDetail(index: Int) {
        root.removeAllViews()
        val r = records[index]
        val content = Ui.col(this)

        val header = Ui.row(this)
        header.addView(Ui.title(this, speciesLabel(r)).apply {
            layoutParams = LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        })
        header.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = getString(R.string.edit)
            setOnClickListener { editDialog(r, index) }
        })
        content.addView(header)

        content.addView(ImageView(this).apply {
            setImageResource(R.drawable.ic_hjort_side)
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, Ui.dp(this@RegistrerteSkuddActivity, 130))
            scaleType = ImageView.ScaleType.FIT_CENTER
        })
        content.addView(Ui.body(this, "${getString(R.string.hunt_position)}: " +
            r.placeName.ifBlank { "—" }))
        content.addView(Ui.body(this, "Dato: ${dateOf(r)}"))
        if (r.distanceM > 0) content.addView(Ui.body(this,
            "${getString(R.string.hunt_hold)}: ${r.distanceM} m"))
        r.ranM?.let { content.addView(Ui.body(this, "${getString(R.string.hunt_ran)} $it m")) }
        content.addView(Ui.body(this, "Utfall: ${r.outcome.label}" +
            (r.followUp?.let { " → ${it.label}" } ?: "")))

        // Innhold scroller, med plass nederst til den faste knapperaden
        content.addView(android.widget.Space(this), LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, Ui.dp(this, 80)))
        root.addView(Ui.scroll(this, content), ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT)

        // Fast knapperad nederst: piler på sidene, OK sentrert (flytter seg ikke)
        val bar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        // Venstre celle: «=>» til eldre (lista er nyest først -> index+1)
        val left = FrameLayout(this)
        if (index < records.size - 1) left.addView(bigArrow("⟸") { renderDetail(index + 1) })
        bar.addView(left, LinearLayout.LayoutParams(0,
            ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        bar.addView(MaterialButton(this).apply {
            text = getString(R.string.ok)
            minWidth = Ui.dp(this@RegistrerteSkuddActivity, 120)
            setOnClickListener { renderList() }
        })
        val right = FrameLayout(this)
        if (index > 0) right.addView(bigArrow("⟹") { renderDetail(index - 1) })
        bar.addView(right, LinearLayout.LayoutParams(0,
            ViewGroup.LayoutParams.WRAP_CONTENT, 1f))

        root.addView(bar, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT,
            Gravity.BOTTOM).apply {
            bottomMargin = Ui.dp(this@RegistrerteSkuddActivity, 24)
            leftMargin = Ui.dp(this@RegistrerteSkuddActivity, 8)
            rightMargin = Ui.dp(this@RegistrerteSkuddActivity, 8)
        })
    }

    private fun bigArrow(glyph: String, onClick: () -> Unit) = MaterialButton(this, null,
        com.google.android.material.R.attr.borderlessButtonStyle).apply {
        text = glyph; textSize = 34f
        setOnClickListener { onClick() }
    }

    /** Enkel redigering av stedsnavn / skuddhold / hvor langt dyret løp. */
    private fun editDialog(r: HuntRecord, index: Int) {
        val col = Ui.col(this, 16)
        val place = EditText(this).apply {
            hint = getString(R.string.hunt_position); setText(r.placeName)
        }
        Ui.capitalize(place)
        val ran = EditText(this).apply {
            hint = getString(R.string.hunt_ran)
            inputType = android.text.InputType.TYPE_CLASS_NUMBER
            setText(r.ranM?.toString() ?: "")
        }
        col.addView(place); col.addView(ran)
        AlertDialog.Builder(this)
            .setTitle(R.string.edit)
            .setView(col)
            .setPositiveButton(R.string.save) { _, _ ->
                val updated = r.copy(placeName = place.text.toString().trim(),
                    ranM = ran.text.toString().toIntOrNull())
                Store.get(this).updateHunt(updated)
                reload()
                val newIdx = records.indexOfFirst { it.id == updated.id }.coerceAtLeast(0)
                renderDetail(newIdx)
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }
}
