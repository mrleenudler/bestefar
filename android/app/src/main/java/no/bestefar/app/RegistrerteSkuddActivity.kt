package no.bestefar.app

import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Locale

/**
 * «Se registrerte skudd» (musingsUI runde 4): liste med vilt-ikon, stedsnavn
 * og dato. Klikk åpner detalj med pil venstre/høyre for å bla — pilene løper
 * ikke rundt (kun én pil vises på første/siste).
 */
class RegistrerteSkuddActivity : AppCompatActivity() {

    private lateinit var content: LinearLayout
    private lateinit var records: List<HuntRecord>
    private val dateFmt = DateTimeFormatter.ofPattern("d. MMMM yyyy", Locale("no"))

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        content = Ui.col(this)
        val scroller = Ui.scroll(this, content)
        Ui.applyInsets(scroller)
        setContentView(scroller)
        records = Store.get(this).allHunts().sortedByDescending { it.ts }
        renderList()
    }

    private fun dateOf(r: HuntRecord) =
        Instant.ofEpochMilli(r.ts).atZone(ZoneId.systemDefault()).toLocalDate().format(dateFmt)

    private fun speciesLabel(r: HuntRecord) =
        if (r.species == Species.ANNET && r.speciesOther.isNotBlank()) r.speciesOther
        else r.species.label

    private fun renderList() {
        content.removeAllViews()
        content.addView(Ui.title(this, getString(R.string.hunt_view_registered)))
        if (records.isEmpty()) {
            content.addView(Ui.hint(this, getString(R.string.hunt_none)))
            return
        }
        records.forEachIndexed { i, r ->
            val row = Ui.row(this).apply {
                setPadding(0, Ui.dp(this@RegistrerteSkuddActivity, 8), 0,
                    Ui.dp(this@RegistrerteSkuddActivity, 8))
                setOnClickListener { renderDetail(i) }
            }
            row.addView(ImageView(this).apply {
                setImageResource(R.drawable.ic_hjort_side)   // arts-ikoner: hjort inntil videre
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
    }

    private fun renderDetail(index: Int) {
        content.removeAllViews()
        val r = records[index]
        content.addView(Ui.title(this, speciesLabel(r)))
        content.addView(ImageView(this).apply {
            setImageResource(R.drawable.ic_hjort_side)
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, Ui.dp(this@RegistrerteSkuddActivity, 140))
            scaleType = ImageView.ScaleType.FIT_CENTER
        })
        content.addView(Ui.body(this, "${getString(R.string.hunt_position)}: " +
            r.placeName.ifBlank { "—" }))
        content.addView(Ui.body(this, "Dato: ${dateOf(r)}"))
        if (r.distanceM > 0) content.addView(Ui.body(this,
            "${getString(R.string.hunt_distance)}: ${r.distanceM} m"))
        r.ranM?.let { content.addView(Ui.body(this,
            "${getString(R.string.hunt_ran)} $it m")) }
        content.addView(Ui.body(this, "Utfall: ${r.outcome.label}" +
            (r.followUp?.let { " → ${it.label}" } ?: "")))

        // Pil venstre/høyre; løper ikke rundt (musingsUI runde 4)
        val nav = Ui.row(this)
        if (index < records.size - 1) {   // eldre finnes (lista er nyest først)
            nav.addView(MaterialButton(this, null,
                com.google.android.material.R.attr.borderlessButtonStyle).apply {
                text = "‹"; textSize = 26f
                setOnClickListener { renderDetail(index + 1) }
            })
        }
        nav.addView(android.widget.Space(this), LinearLayout.LayoutParams(0, 1, 1f))
        nav.addView(MaterialButton(this).apply {
            text = getString(R.string.ok)
            setOnClickListener { renderList() }
        })
        nav.addView(android.widget.Space(this), LinearLayout.LayoutParams(0, 1, 1f))
        if (index > 0) {   // nyere finnes
            nav.addView(MaterialButton(this, null,
                com.google.android.material.R.attr.borderlessButtonStyle).apply {
                text = "›"; textSize = 26f
                setOnClickListener { renderDetail(index - 1) }
            })
        }
        content.addView(nav, Ui.matchWrap(16, this))
    }
}
