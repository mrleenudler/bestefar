package no.bestefar.app

import android.content.res.Configuration
import android.graphics.Color
import android.os.Bundle
import android.text.InputType
import android.view.Gravity
import android.view.ViewGroup
import android.widget.CheckBox
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.widget.ImageViewCompat
import com.google.android.material.button.MaterialButton
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Locale

/**
 * «Se registrerte skudd» (musingsUI runde 6): liste med vilt-ikon, stedsnavn,
 * dato (nyeste samme-dag øverst). Klikk-og-hold merker flere → «Slett alle»
 * nederst t.h. Detalj har store piler (fjernes i endene, men OK står fast),
 * Rediger som endrer all info, og tilbake-knapp går til lista.
 */
class RegistrerteSkuddActivity : AppCompatActivity() {

    private lateinit var root: FrameLayout
    private lateinit var records: MutableList<HuntRecord>
    private val dateFmt = DateTimeFormatter.ofPattern("d. MMMM yyyy", Locale("no"))
    private var selectionMode = false
    private val selected = mutableSetOf<String>()
    private var detailIndex: Int? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        root = FrameLayout(this)
        Ui.applyInsets(root)
        setContentView(root)
        reload()
        renderList()
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                when {
                    detailIndex != null -> { detailIndex = null; renderList() }
                    selectionMode -> exitSelection()
                    else -> finish()
                }
            }
        })
    }

    private fun reload() {
        // Nyeste samme-dag øverst: sorter på dato, så opprettelsestid (runde 6)
        records = Store.get(this).allHunts()
            .sortedWith(compareByDescending<HuntRecord> { it.ts }.thenByDescending { it.created })
            .toMutableList()
    }

    private fun night() = (resources.configuration.uiMode and
        Configuration.UI_MODE_NIGHT_MASK) == Configuration.UI_MODE_NIGHT_YES

    private fun silTint() = if (night())
        Ui.themeColor(this, android.R.attr.textColorPrimary) else Color.BLACK

    private fun dateOf(r: HuntRecord) =
        Instant.ofEpochMilli(r.ts).atZone(ZoneId.systemDefault()).toLocalDate().format(dateFmt)

    private fun speciesLabel(r: HuntRecord) =
        if (r.species == Species.ANNET && r.speciesOther.isNotBlank()) r.speciesOther
        else r.species.label

    private fun silhouette(size: Int) = ImageView(this).apply {
        setImageResource(R.drawable.ic_hjort_side)
        ImageViewCompat.setImageTintList(this,
            android.content.res.ColorStateList.valueOf(silTint()))
        layoutParams = LinearLayout.LayoutParams(Ui.dp(this@RegistrerteSkuddActivity, size),
            Ui.dp(this@RegistrerteSkuddActivity, size))
    }

    // ---------- Liste ----------

    private fun renderList() {
        root.removeAllViews()
        detailIndex = null
        val content = Ui.col(this)
        content.addView(Ui.title(this, getString(R.string.hunt_view_registered)))

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
                    } else { detailIndex = i; renderDetail(i) }
                }
                setOnLongClickListener {
                    if (!selectionMode) { selectionMode = true; selected.add(r.id); renderList() }
                    true
                }
            }
            row.addView(silhouette(36))
            row.addView(TextView(this).apply {
                text = "  ${speciesLabel(r)} · ${r.placeName.ifBlank { "—" }} · ${dateOf(r)}"
                textSize = 16f
            })
            content.addView(row)
        }
        content.addView(android.widget.Space(this), LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, Ui.dp(this, 72)))
        root.addView(Ui.scroll(this, content), ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT)

        if (selectionMode) {
            // «Slett alle» nederst t.h. (musingsUI runde 6)
            root.addView(MaterialButton(this).apply {
                text = getString(R.string.delete_all)
                setOnClickListener { confirmDeleteAll() }
            }, FrameLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM or Gravity.END).apply {
                bottomMargin = Ui.dp(this@RegistrerteSkuddActivity, 16)
                rightMargin = Ui.dp(this@RegistrerteSkuddActivity, 16)
            })
            root.addView(MaterialButton(this, null,
                com.google.android.material.R.attr.borderlessButtonStyle).apply {
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

    private fun confirmDeleteAll() {
        AlertDialog.Builder(this)
            .setMessage(R.string.delete_all_confirm)
            .setPositiveButton(R.string.delete_all) { _, _ ->
                Store.get(this).deleteHunts(selected.toSet()); reload(); exitSelection()
            }
            .setNegativeButton(R.string.cancel) { _, _ -> finish() }   // -> hovedsiden
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
            setOnClickListener { editDialog(r) }
        })
        content.addView(header)

        content.addView(silhouette(120).apply {
            (layoutParams as LinearLayout.LayoutParams).width =
                ViewGroup.LayoutParams.MATCH_PARENT
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
        if (r.outcome == Outcome.DOEDELIG) content.addView(Ui.hint(this,
            if (r.fellingSuccess) "Felling vellykket" else "Felling mislykket"))

        content.addView(android.widget.Space(this), LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, Ui.dp(this, 90)))
        root.addView(Ui.scroll(this, content), ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT)

        // Fast knapperad: piler på sidene (fjernes i endene), OK sentrert
        val bar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
        }
        val left = FrameLayout(this)
        if (index < records.size - 1) left.addView(bigArrow("‹") { renderDetail(index + 1) })
        bar.addView(left, LinearLayout.LayoutParams(0,
            ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        bar.addView(MaterialButton(this).apply {
            text = getString(R.string.ok)
            minWidth = Ui.dp(this@RegistrerteSkuddActivity, 120)
            setOnClickListener { renderList() }
        })
        val right = FrameLayout(this)
        if (index > 0) right.addView(bigArrow("›") { renderDetail(index - 1) })
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

    /** Store, høye piler — samme stil brukes overalt hvor vi blar (runde 6). */
    private fun bigArrow(glyph: String, onClick: () -> Unit) = MaterialButton(this, null,
        com.google.android.material.R.attr.borderlessButtonStyle).apply {
        text = glyph; textSize = 40f; scaleY = 1.7f
        setOnClickListener { onClick() }
    }

    /** Rediger all info, inkl. utfall/ettersøk/ikke funnet (musingsUI runde 6). */
    private fun editDialog(r: HuntRecord) {
        val col = Ui.col(this, 16)
        val place = EditText(this).apply {
            hint = getString(R.string.hunt_position); setText(r.placeName)
        }
        Ui.capitalize(place)
        val hold = EditText(this).apply {
            hint = getString(R.string.hunt_hold)
            inputType = InputType.TYPE_CLASS_NUMBER
            setText(if (r.distanceM > 0) r.distanceM.toString() else "")
        }
        val ran = EditText(this).apply {
            hint = getString(R.string.hunt_ran)
            inputType = InputType.TYPE_CLASS_NUMBER
            setText(r.ranM?.toString() ?: "")
        }
        col.addView(place); col.addView(hold); col.addView(ran)

        var outcome = r.outcome
        var notFound = r.followUp == FollowUp.IKKE_GJENFUNNET
        val outcomeRow = Ui.row(this)
        fun renderOutcome() {
            outcomeRow.removeAllViews()
            Outcome.entries.forEach { o ->
                outcomeRow.addView(Ui.choiceButton(this, o.label, outcome == o) {
                    outcome = o; renderOutcome()
                }.apply { layoutParams = LinearLayout.LayoutParams(0,
                    ViewGroup.LayoutParams.WRAP_CONTENT, 1f) })
            }
        }
        renderOutcome()
        col.addView(outcomeRow)
        val notFoundBox = CheckBox(this).apply {
            text = getString(R.string.hunt_not_found); isChecked = notFound
        }
        col.addView(notFoundBox)

        AlertDialog.Builder(this)
            .setTitle(R.string.edit)
            .setView(androidx.core.widget.NestedScrollView(this).apply { addView(col) })
            .setPositiveButton(R.string.save) { _, _ ->
                val updated = r.copy(
                    placeName = place.text.toString().trim(),
                    distanceM = hold.text.toString().toIntOrNull() ?: 0,
                    ranM = ran.text.toString().toIntOrNull(),
                    outcome = outcome,
                    followUp = if (notFoundBox.isChecked) FollowUp.IKKE_GJENFUNNET else r.followUp)
                Store.get(this).updateHunt(updated)
                reload()
                val idx = records.indexOfFirst { it.id == updated.id }.coerceAtLeast(0)
                renderDetail(idx)
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }
}
