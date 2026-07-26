package no.bestefar.app

import android.graphics.Color
import android.view.Gravity
import android.view.ViewGroup
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.TableLayout
import android.widget.TableRow
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import com.google.android.material.button.MaterialButton
import com.google.android.material.button.MaterialButtonToggleGroup
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import kotlin.math.abs

/** Felles skjelett for programmatiske fragmenter. */
abstract class RebuildFragment : androidx.fragment.app.Fragment() {
    protected lateinit var content: LinearLayout

    override fun onCreateView(inflater: android.view.LayoutInflater,
                              container: ViewGroup?,
                              savedInstanceState: android.os.Bundle?): android.view.View {
        content = Ui.col(requireContext())
        return Ui.scroll(requireContext(), content)
    }

    override fun onResume() {
        super.onResume()
        rebuild()
    }

    protected abstract fun rebuild()
}

/**
 * Innsikt (spec §5): kompetanseoversikt (primær) og kapabilitetskart
 * (sekundær) bak segmentkontroll. Skjult til første økt i sesongen.
 * Frekvens og ren gevinstramme; farge fra felles forsvarlighetskriterium.
 */
class InnsiktFragment : RebuildFragment() {

    private var mode = 0                      // 0 = kompetanse, 1 = kart
    private var species = Species.ELG
    private var angleIdx = 0                  // indeks i Angle.entries
    private var holdM = 100

    private val distances = listOf(50, 100, 150, 200, 300)

    override fun rebuild() {
        val a = requireActivity()
        val store = Store.get(a)
        content.removeAllViews()
        content.addView(Ui.title(a, getString(R.string.tab_innsikt)))

        val evidence = store.currentSeasonSeries().filter { it.countsInEvidence }
        if (evidence.isEmpty()) {
            // Innsikt låses opp før serier er skutt (musingsUI runde 4):
            // vis hva flaten gjør, med en nøytral demo-figur.
            content.addView(Ui.body(a, getString(R.string.innsikt_preview)))
            content.addView(AnimalView(a).apply {
                layoutParams = LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, Ui.dp(a, 150))
                tintColor = Color.GRAY
                label = getString(R.string.innsikt_not_tested)
            })
            merStatistikkButton(store)
            return
        }

        // Segmentkontroll
        val toggle = MaterialButtonToggleGroup(a).apply {
            isSingleSelection = true
            isSelectionRequired = true
        }
        val btnComp = MaterialButton(a, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            id = ViewGroup.generateViewId()
            text = getString(R.string.innsikt_competence)
        }
        val btnMap = MaterialButton(a, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            id = ViewGroup.generateViewId()
            text = getString(R.string.innsikt_map)
        }
        toggle.addView(btnComp); toggle.addView(btnMap)
        toggle.check(if (mode == 0) btnComp.id else btnMap.id)
        toggle.addOnButtonCheckedListener { _, id, checked ->
            if (checked) { mode = if (id == btnComp.id) 0 else 1; rebuild() }
        }
        content.addView(toggle)

        // Artsvelger (norsk storvilt; «annet» holdes utenfor analyser)
        val chips = ChipGroup(a).apply { isSingleSelection = true }
        Species.entries.filter { it != Species.ANNET }.forEach { s ->
            chips.addView(Chip(a).apply {
                text = s.label; isCheckable = true; isChecked = s == species
                setOnClickListener { species = s; rebuild() }
            })
        }
        content.addView(chips)

        if (mode == 0) buildCompetence(evidence) else buildMap(evidence)

        merStatistikkButton(store)
    }

    /** «Mer statistikk» nederst i Innsikt (flyttet fra menyen, musingsUI r4). */
    private fun merStatistikkButton(store: Store) {
        val a = requireActivity()
        content.addView(MaterialButton(a, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            text = getString(R.string.menu_more_stats)
            layoutParams = Ui.matchWrap(20, a)
            setOnClickListener {
                val evid = store.currentSeasonSeries().filter { it.countsInEvidence }
                val sigma = Stats.sigmaCmAt100(evid)
                val msg = if (sigma == null) getString(R.string.stats_none)
                else "σ (100 m-ekvivalent): %.1f cm\nR95: %.1f cm\nSpredning: %.2f MOA\n\nMålt om siktepunktet (%d skudd)."
                    .format(sigma, Stats.r95Cm(sigma), Stats.moa(sigma), Stats.shotCount(evid))
                AlertDialog.Builder(a).setTitle(R.string.menu_more_stats)
                    .setMessage(msg).setPositiveButton(R.string.ok, null).show()
            }
        })
    }

    // ---------- Kompetanseoversikt ----------

    private fun buildCompetence(evidence: List<SeriesRecord>) {
        val a = requireActivity()
        val store = Store.get(a)
        val angle = Angle.entries[angleIdx]
        val sigma = Stats.sigmaCmAt100(evidence)!!
        val n = Stats.shotCount(evidence)
        val radius = Stats.lethalRadiusCm(species, angle) ?: return
        val p = Stats.pLethal(sigma, holdM, radius)
        val (pLo, pHi) = Stats.pLethalSpan(sigma, n, holdM, radius)
        val color = Stats.rateColor(p, store.rateLimit)

        // Dyrefigur (placeholder-silhuett) med rotasjonspiler for vinkling
        val figRow = Ui.row(a).apply { gravity = Gravity.CENTER }
        figRow.addView(MaterialButton(a, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = "‹"; textSize = 28f
            contentDescription = "Forrige vinkling"
            setOnClickListener {
                angleIdx = (angleIdx + Angle.entries.size - 1) % Angle.entries.size
                rebuild()
            }
        })
        val animal = AnimalView(a).apply {
            layoutParams = LinearLayout.LayoutParams(0, Ui.dp(a, 160), 1f)
            tintColor = color
            holdScale = 1f - (holdM - 50) / 1000f   // subtil skalering (spec §5)
            facingLeft = angle != Angle.BAK
            label = angle.label
        }
        figRow.addView(animal)
        figRow.addView(MaterialButton(a, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = "›"; textSize = 28f
            contentDescription = "Neste vinkling"
            setOnClickListener { angleIdx = (angleIdx + 1) % Angle.entries.size; rebuild() }
        })
        content.addView(figRow)

        // Holdvelger
        val holds = ChipGroup(a).apply { isSingleSelection = true }
        distances.forEach { d ->
            holds.addView(Chip(a).apply {
                text = "$d m"; isCheckable = true; isChecked = holdM == d
                setOnClickListener { holdM = d; rebuild() }
            })
        }
        content.addView(holds)

        // Frekvensbudskap i ren gevinstramme (spec §5/§8)
        val freqRow = Ui.row(a)
        freqRow.addView(TextView(a).apply {
            text = getString(R.string.innsikt_fells, Stats.freqTextWithSpan(p, pLo, pHi, n))
            textSize = 22f
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        })
        freqRow.addView(ImageButton(a).apply {
            setImageResource(R.drawable.ic_info)
            background = null
            contentDescription = "Forklaring"
            setOnClickListener {
                AlertDialog.Builder(a).setMessage(R.string.innsikt_info)
                    .setPositiveButton(R.string.ok, null).show()
            }
        })
        content.addView(freqRow)

        // Maks forsvarlig hold per stilling (spec §5)
        content.addView(Ui.section(a, getString(R.string.innsikt_max_hold)))
        Position.hoved.forEach { pos ->
            val posSeries = evidence.filter { it.position == pos }
            val line = if (posSeries.isEmpty()) {
                "${pos.label}: ${getString(R.string.innsikt_not_tested)}"
            } else {
                val sp = Stats.sigmaCmAt100(posSeries)!!
                "${pos.label}: ~${Stats.maxHoldM(sp, radius, store.rateLimit)} m"
            }
            content.addView(Ui.body(a, line))
        }
        content.addView(Ui.hint(a, getString(R.string.innsikt_moving_note)))
    }

    // ---------- Kapabilitetskart ----------

    private fun buildMap(evidence: List<SeriesRecord>) {
        val a = requireActivity()
        val store = Store.get(a)
        // Kartet bruker bredside-radius; vinkling velges i kompetansevisningen
        val radius = Stats.lethalRadiusCm(species, Angle.SIDE) ?: return

        val table = TableLayout(a).apply {
            isStretchAllColumns = true
            layoutParams = Ui.matchWrap(12, a)
        }
        val header = TableRow(a)
        header.addView(cell(a, "", Color.TRANSPARENT))
        distances.forEach { d -> header.addView(cell(a, "$d m", Color.TRANSPARENT)) }
        table.addView(header)

        Position.hoved.forEach { pos ->
            val rowSeries = evidence.filter { it.position == pos }
            val sigmaPos = Stats.sigmaCmAt100(rowSeries)
            val row = TableRow(a)
            row.addView(cell(a, pos.label, Color.TRANSPARENT))
            distances.forEach { d ->
                if (sigmaPos == null) {
                    // «ikke testet» som inngang til øvelsesmotoren (spec §5)
                    row.addView(cell(a, getString(R.string.innsikt_not_tested),
                        Color.argb(40, 128, 128, 128)).apply {
                        setOnClickListener {
                            store.practicePosition = pos
                            store.distanceM = d
                            Toast.makeText(a, "Øvelse satt opp: ${pos.label}, $d m",
                                Toast.LENGTH_SHORT).show()
                        }
                    })
                } else {
                    val n = Stats.shotCount(rowSeries)
                    val p = Stats.pLethal(sigmaPos, d, radius)
                    val (pLo, pHi) = Stats.pLethalSpan(sigmaPos, n, d, radius)
                    val measured = rowSeries.any { abs(it.distanceM - d) <= d * 0.15 }
                    val txt = (if (measured) "" else "~") +
                        Stats.freqTextWithSpan(p, pLo, pHi, n)
                    row.addView(cell(a, txt, Stats.rateColor(p, store.rateLimit)).apply {
                        if (!measured) alpha = 0.7f   // ekstrapolert (stiplet ramme TODO)
                        setOnClickListener { cellDetail(pos, d, rowSeries, p, pLo, pHi) }
                    })
                }
            }
            table.addView(row)
        }
        content.addView(table)
    }

    private fun cell(a: android.app.Activity, txt: String, bg: Int): TextView =
        TextView(a).apply {
            text = txt
            gravity = Gravity.CENTER
            textSize = 12f
            setTextColor(Color.BLACK)
            setPadding(Ui.dp(a, 4), Ui.dp(a, 12), Ui.dp(a, 4), Ui.dp(a, 12))
            setBackgroundColor(bg)
            if (bg == Color.TRANSPARENT) setTextColor(
                Ui.themeColor(a, android.R.attr.textColorPrimary))
        }

    /** Celledetalj: spenn, antall skudd, dato for siste måling, serier (spec §5). */
    private fun cellDetail(pos: Position, d: Int, series: List<SeriesRecord>,
                           p: Double, pLo: Double, pHi: Double) {
        val a = requireActivity()
        val last = series.maxByOrNull { it.ts }
        val fmt = DateTimeFormatter.ofPattern("d.M.yyyy")
        val lastStr = last?.let {
            Instant.ofEpochMilli(it.ts).atZone(ZoneId.systemDefault()).format(fmt)
        } ?: "—"
        AlertDialog.Builder(a)
            .setTitle("${pos.label} · $d m · ${species.label}")
            .setMessage(
                "Dødelige treff: ${Stats.freqText(p)}\n" +
                "Spenn: ${Stats.freqText(pLo)} til ${Stats.freqText(pHi)}\n" +
                "Skudd: ${Stats.shotCount(series)} i ${series.size} serier\n" +
                "Siste måling: $lastStr")
            .setPositiveButton(R.string.ok, null)
            .show()
    }
}
