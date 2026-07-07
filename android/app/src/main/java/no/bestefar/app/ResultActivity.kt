package no.bestefar.app

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton

/** Resultatskjerm: poengliste + sum. OK -> tilbake til Start (kravspec §7). */
class ResultActivity : AppCompatActivity() {
    companion object {
        const val EXTRA_STATUS = "status"
        const val EXTRA_SUM_DEC = "sum_dec"
        const val EXTRA_SUM_INT = "sum_int"
        const val EXTRA_N_HITS = "n_hits"
        const val EXTRA_CONFIDENCE = "confidence"
        const val EXTRA_DECIMALS = "decimals"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_result)

        val status = intent.getIntExtra(EXTRA_STATUS, BestefarCore.ERROR_INTERNAL)
        val text = findViewById<TextView>(R.id.resultText)
        if (status == BestefarCore.OK) {
            val decimals = intent.getDoubleArrayExtra(EXTRA_DECIMALS) ?: doubleArrayOf()
            val sumDec = intent.getDoubleExtra(EXTRA_SUM_DEC, 0.0)
            val sumInt = intent.getIntExtra(EXTRA_SUM_INT, 0)
            val conf = intent.getDoubleExtra(EXTRA_CONFIDENCE, 0.0)
            text.text = buildString {
                appendLine(getString(R.string.result_hits,
                                     intent.getIntExtra(EXTRA_N_HITS, 0)))
                appendLine(decimals.joinToString("  ") { "%.1f".format(it) })
                appendLine()
                appendLine(getString(R.string.result_sum, sumDec, sumInt))
                appendLine(getString(R.string.result_confidence, conf))
            }
        } else {
            // Lav konfidens / avvist -> kandidat for feilinnsending (kravspec §5.2)
            text.text = getString(R.string.result_rejected, status)
        }

        findViewById<MaterialButton>(R.id.okButton).setOnClickListener { finish() }
    }
}
