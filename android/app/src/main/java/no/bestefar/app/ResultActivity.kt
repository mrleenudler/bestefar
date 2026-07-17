package no.bestefar.app

import android.content.Intent
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
import kotlin.math.sqrt

/**
 * Resultatkort (spec §2): skudd plottet på skivegjengivelse, poengsum,
 * langsiktig snitt/spredning, klikk-forslag (kun når skjelnbart fra støy),
 * korrigering av skuddmerker, og teller/teller-ikke-status.
 *
 * Flyt: stillingsprompt (med mindre manuell/øvelsesmodus) → dagsbekreftelse
 * av våpen (første serie den dagen) → lagring → kort → ev. samtykkeprompt.
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

        // Gjenskaping (rotasjon): ikke prompt/lagre på nytt
        savedInstanceState?.getString("recordId")?.let { id ->
            record = store.allSeries().firstOrNull { it.id == id }
            if (record != null) { render(); return }
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
                val r = SeriesRecord(
                    id = Store.newId(),
                    ts = System.currentTimeMillis(),
                    weaponId = w?.id,
                    ammoName = w?.ammoName ?: "",
                    distanceM = store.distanceM,
                    position = pos,
                    modifier = mod,
                    shots = shots,
                )
                store.addSeries(r)
                record = r
                render()
                Dialogs.maybeResearchConsent(this, store)
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
        content.addView(okButton())
    }

    private fun render() {
        val r = record ?: return
        content.removeAllViews()

        val target = TargetView(this).apply {
            hits = r.shots
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, Ui.dp(this@ResultActivity, 320))
        }
        content.addView(target)

        // Sum + korrigerings-blyant (spec §2)
        val sumRow = Ui.row(this)
        sumRow.addView(TextView(this).apply {
            text = "%.1f  (%d)".format(r.sumDecimal, r.sumInteger)
            textSize = 30f
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        })
        sumRow.addView(ImageButton(this).apply {
            setImageResource(R.drawable.ic_edit)
            background = null
            contentDescription = getString(R.string.result_edit)
            setOnClickListener { Dialogs.correctionDialog(this@ResultActivity, r, store) { render() } }
        })
        content.addView(sumRow)

        content.addView(Ui.body(this,
            r.shots.joinToString("   ") { "%.1f".format(it.decimal) }))
        val w = store.weapons().firstOrNull { it.id == r.weaponId }
        content.addView(Ui.hint(this,
            "${r.position.label} (${r.modifier.label}) · ${r.distanceM} m · ${w?.name ?: "—"}"))

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
        val click = w?.clickValueCm
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

        val btnRow = Ui.row(this).apply { gravity = Gravity.END }
        btnRow.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = getString(R.string.result_end_session)
            setOnClickListener {
                startActivity(Intent(this@ResultActivity, SummaryActivity::class.java))
                finish()
            }
        })
        btnRow.addView(okButton())
        content.addView(btnRow)
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        record?.let { outState.putString("recordId", it.id) }
    }

    private fun okButton() = MaterialButton(this).apply {
        text = getString(R.string.ok)
        layoutParams = LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { topMargin = Ui.dp(this@ResultActivity, 8) }
        setOnClickListener { finish() }
    }
}
