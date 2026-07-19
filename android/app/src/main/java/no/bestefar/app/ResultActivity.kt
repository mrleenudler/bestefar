package no.bestefar.app

import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.sqrt

/**
 * Resultatkort (musingsUI): skive med treffene markert; poengene i stigende
 * rekkefølge på høyre side, blyant per poeng for korrigering; «Ikke lagre»
 * til venstre og «OK» til høyre under skiven. Serien lagres først ved OK.
 */
class ResultActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_STATUS = "status"
        const val EXTRA_SUM_DEC = "sum_dec"
        const val EXTRA_SUM_INT = "sum_int"
        const val EXTRA_N_HITS = "n_hits"
        const val EXTRA_CONFIDENCE = "confidence"
        const val EXTRA_DECIMALS = "decimals"
        const val EXTRA_INTEGERS = "integers"
        const val EXTRA_RREL = "r_rel"
        const val EXTRA_THETA = "theta"
    }

    private lateinit var store: Store
    private lateinit var content: LinearLayout
    private var record: SeriesRecord? = null
    private var saved = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = Store.get(this)
        content = Ui.col(this)
        setContentView(Ui.scroll(this, content))

        val status = intent.getIntExtra(EXTRA_STATUS, BestefarCore.ERROR_INTERNAL)
        if (status != BestefarCore.OK) {
            renderRejected(status)
            return
        }

        // Gjenskaping (rotasjon): allerede lagret -> bare vis
        savedInstanceState?.getString("recordId")?.let { id ->
            record = store.allSeries().firstOrNull { it.id == id }
            if (record != null) { saved = true; render(); return }
        }

        val decimals = intent.getDoubleArrayExtra(EXTRA_DECIMALS) ?: doubleArrayOf()
        val integers = intent.getIntArrayExtra(EXTRA_INTEGERS)
            ?: decimals.map { it.toInt() }.toIntArray()
        val rrel = intent.getDoubleArrayExtra(EXTRA_RREL) ?: DoubleArray(decimals.size)
        val theta = intent.getDoubleArrayExtra(EXTRA_THETA) ?: DoubleArray(decimals.size)
        val shots = decimals.indices.map {
            Shot(decimals[it], integers[it], rrel[it], theta[it])
        }

        resolvePosition { pos, mod ->
            Dialogs.weaponDayConfirm(this, store) {
                val w = store.selectedWeapon()
                record = SeriesRecord(
                    id = Store.newId(),
                    ts = System.currentTimeMillis(),
                    weaponId = w?.id,
                    ammoName = w?.ammoName ?: "",
                    distanceM = store.distanceM,
                    position = pos,
                    modifier = mod,
                    shots = shots,
                )
                render()
            }
        }
    }

    /** Stillingsprompt etter scan; hoppes over i manuell/øvelsesmodus (spec §2). */
    private fun resolvePosition(onDone: (Position, PosModifier) -> Unit) {
        val practice = store.practicePosition
        when {
            practice != null -> onDone(practice, store.lastModifier(practice))
            store.manualPosition -> onDone(store.currentPosition, store.currentModifier)
            else -> Dialogs.positionSheet(this, store, onDone)
        }
    }

    private fun renderRejected(status: Int) {
        content.removeAllViews()
        content.addView(Ui.title(this, getString(R.string.app_name)))
        content.addView(Ui.body(this, getString(R.string.result_rejected, status)))
        // Kandidat for feilanalysekanalen (kravspec §5.2); kø-kobling kommer
        content.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            text = getString(R.string.result_send_fail)
            layoutParams = Ui.matchWrap(16, this@ResultActivity)
            setOnClickListener {
                Toast.makeText(this@ResultActivity,
                    getString(R.string.summary_queue, 1), Toast.LENGTH_SHORT).show()
                isEnabled = false
            }
        })
        content.addView(MaterialButton(this).apply {
            text = getString(R.string.ok)
            layoutParams = Ui.matchWrap(8, this@ResultActivity)
            setOnClickListener { finish() }
        })
    }

    private fun render() {
        val r = record ?: return
        content.removeAllViews()

        // Skive + poengliste (stigende) på høyre side
        val mainRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val target = TargetView(this).apply {
            hits = r.shots
            layoutParams = LinearLayout.LayoutParams(0,
                Ui.dp(this@ResultActivity, 300), 1f)
        }
        mainRow.addView(target)

        val scoreCol = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(Ui.dp(this@ResultActivity, 8), 0, 0, 0)
        }
        val ordered = r.shots.withIndex().sortedBy { it.value.decimal }
        ordered.forEach { (idx, shot) ->
            val row = Ui.row(this)
            row.addView(TextView(this).apply {
                text = "%.1f".format(shot.decimal)
                textSize = 19f
                minWidth = Ui.dp(this@ResultActivity, 48)
            })
            row.addView(ImageButton(this).apply {
                setImageResource(R.drawable.ic_edit)
                background = null
                contentDescription = getString(R.string.result_edit)
                setOnClickListener {
                    Dialogs.shotEdit(this@ResultActivity, shot.decimal) { v ->
                        r.shots = r.shots.mapIndexed { i, s ->
                            if (i == idx) s.copy(decimal = v,
                                integer = floor(v).toInt().coerceAtMost(10)) else s
                        }
                        r.corrected = true
                        if (saved) store.updateSeries(r)
                        render()
                    }
                }
            })
            scoreCol.addView(row)
        }
        mainRow.addView(scoreCol)
        content.addView(mainRow)

        content.addView(TextView(this).apply {
            text = "%.1f  (%d)".format(r.sumDecimal, r.sumInteger)
            textSize = 28f
        })
        val w = store.weapons().firstOrNull { it.id == r.weaponId }
        content.addView(Ui.hint(this,
            "${r.position.label} (${r.modifier.label}) · ${r.distanceM} m · " +
            (w?.shownName ?: "—")))

        // Status i evidensgrunnlaget (benk teller ikke, spec §2/§8)
        content.addView(Ui.body(this, getString(
            if (r.countsInEvidence) R.string.result_counts else R.string.result_counts_not)))

        // Langsiktig tilstand — serien faller inn i bildet, ingen dom (spec §5)
        val history = store.allSeries().filter {
            it.id != r.id && it.position == r.position &&
                it.distanceM == r.distanceM && it.countsInEvidence
        }
        if (history.isEmpty() || !r.countsInEvidence) {
            content.addView(Ui.hint(this, getString(R.string.result_long_term_none)))
        } else {
            val sums = history.map { it.sumDecimal }
            val avg = sums.average()
            val sd = if (sums.size > 1)
                sqrt(sums.sumOf { (it - avg) * (it - avg) } / (sums.size - 1)) else 0.0
            content.addView(Ui.body(this, getString(R.string.result_long_term,
                r.position.label, r.distanceM, avg, sd, history.size)))
        }

        // Klikk-forslag (spec §2): kun når offset er skjelnbart fra støy
        val click = store.clickCmFor(w)
        if (click == null) {
            content.addView(Ui.hint(this, getString(R.string.result_click_missing)))
        } else {
            val sug = Stats.clickSuggestion(r.shots, r.distanceM, click)
            if (sug == null) {
                content.addView(Ui.body(this, getString(R.string.result_click_noise)))
            } else {
                val (right, up) = sug
                val parts = mutableListOf<String>()
                if (right != 0) parts.add("${abs(right)} klikk ${if (right > 0) "høyre" else "venstre"}")
                if (up != 0) parts.add("${abs(up)} klikk ${if (up > 0) "opp" else "ned"}")
                // TODO: anvendte justeringer logges som hendelser på oppsettet
                content.addView(Ui.body(this,
                    getString(R.string.result_click, parts.joinToString(", "))))
            }
        }

        // «Ikke lagre» venstre, «OK» høyre (musingsUI)
        val btnRow = Ui.row(this)
        btnRow.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = getString(R.string.result_discard)
            setOnClickListener { finish() }
        })
        btnRow.addView(android.widget.Space(this),
            LinearLayout.LayoutParams(0, 1, 1f))
        btnRow.addView(MaterialButton(this).apply {
            text = getString(R.string.ok)
            minWidth = Ui.dp(this@ResultActivity, 120)
            setOnClickListener { saveAndFinish() }
        })
        content.addView(btnRow, Ui.matchWrap(12, this))
    }

    private fun saveAndFinish() {
        val r = record ?: return finish()
        if (!saved) {
            store.addSeries(r)
            saved = true
        }
        val afterConsent = { Dialogs.maybeResearchConsent(this, store) { finish() } }
        if (r.corrected && !r.sendToFailChannel) {
            // Korrigerte analyser tilbys feilanalysekanalen (mikrosamtykke, spec §2)
            Dialogs.failChannelConsent(this) { yes ->
                if (yes) { r.sendToFailChannel = true; store.updateSeries(r) }
                afterConsent()
            }
        } else {
            afterConsent()
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        if (saved) record?.let { outState.putString("recordId", it.id) }
    }
}
