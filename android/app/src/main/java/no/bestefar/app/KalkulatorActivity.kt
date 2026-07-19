package no.bestefar.app

import android.os.Bundle
import android.text.Editable
import android.text.InputType
import android.text.TextWatcher
import android.view.Gravity
import android.view.ViewGroup
import android.widget.CheckBox
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton
import com.google.android.material.button.MaterialButtonToggleGroup

/**
 * Optikk-kalkulator (musingsUI-runde 2): «Fra» <verdi> <enhet> → «Til»
 * <read-only omregning> <enhet>, ved valgt avstand. SMOA-valg påvirker
 * MOA-omregningen. Fullskjerm med X i hjørnet.
 */
class KalkulatorActivity : AppCompatActivity() {

    private var fromUnit = "CM"
    private var toUnit = "MOA"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = FrameLayout(this)
        Ui.applyInsets(root)
        val content = Ui.col(this)

        content.addView(Ui.title(this, getString(R.string.kalkulator)))

        // Avstand (trengs for cm-omregning)
        val distRow = Ui.row(this)
        distRow.addView(TextView(this).apply {
            text = "${getString(R.string.kalk_distance)}: "
            textSize = 16f
        })
        val distance = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER
            minWidth = Ui.dp(this@KalkulatorActivity, 80)
            setText("100")
        }
        distRow.addView(distance)
        content.addView(distRow)

        fun unitToggle(initial: String, onChange: (String) -> Unit): MaterialButtonToggleGroup {
            val toggle = MaterialButtonToggleGroup(this).apply {
                isSingleSelection = true; isSelectionRequired = true
            }
            val buttons = listOf("CM" to "cm", "MOA" to "MOA", "MRAD" to "mrad").map { (k, l) ->
                MaterialButton(this, null,
                    com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
                    id = ViewGroup.generateViewId(); text = l; tag = k
                }
            }
            buttons.forEach { toggle.addView(it) }
            toggle.check(buttons.first { it.tag == initial }.id)
            toggle.addOnButtonCheckedListener { _, id, checked ->
                if (checked) onChange(buttons.first { it.id == id }.tag as String)
            }
            return toggle
        }

        content.addView(Ui.section(this, getString(R.string.kalk_from)))
        val fromValue = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER or
                InputType.TYPE_NUMBER_FLAG_DECIMAL or InputType.TYPE_NUMBER_FLAG_SIGNED
            hint = getString(R.string.kalk_value)
        }
        content.addView(fromValue)

        content.addView(Ui.section(this, getString(R.string.kalk_to)))
        val toValue = EditText(this).apply {
            isEnabled = false   // viser omregningen; inntasting deaktivert
            textSize = 20f
        }
        content.addView(toValue)

        val smoa = CheckBox(this).apply { text = getString(R.string.optic_smoa) }
        content.addView(smoa, Ui.matchWrap(12, this))

        fun recompute() {
            val d = distance.text.toString().toIntOrNull() ?: 100
            val v = fromValue.text.toString().replace(',', '.').toDoubleOrNull()
            if (v == null || d <= 0) { toValue.setText(""); return }
            val moaCm = if (smoa.isChecked) OpticProfile.SMOA_CM_PER_100M
                else OpticProfile.MOA_CM_PER_100M
            val cmPerMoa = moaCm * d / 100.0
            val cmPerMrad = 10.0 * d / 100.0
            val cm = when (fromUnit) {
                "MOA" -> v * cmPerMoa
                "MRAD" -> v * cmPerMrad
                else -> v
            }
            val out = when (toUnit) {
                "MOA" -> "%.2f MOA".format(cm / cmPerMoa)
                "MRAD" -> "%.2f mrad".format(cm / cmPerMrad)
                else -> "%.1f cm".format(cm)
            }
            toValue.setText(out)
        }

        // Enhetsvelgerne settes inn etter feltene sine
        val fromToggle = unitToggle(fromUnit) { fromUnit = it; recompute() }
        content.addView(fromToggle, content.indexOfChild(fromValue) + 1)
        val toToggle = unitToggle(toUnit) { toUnit = it; recompute() }
        content.addView(toToggle, content.indexOfChild(toValue) + 1)

        val watcher = object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
            override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
            override fun afterTextChanged(s: Editable?) = recompute()
        }
        fromValue.addTextChangedListener(watcher)
        distance.addTextChangedListener(watcher)
        smoa.setOnCheckedChangeListener { _, _ -> recompute() }
        recompute()

        root.addView(Ui.scroll(this, content), ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT)
        // X i hjørnet (musingsUI)
        root.addView(ImageButton(this).apply {
            setImageResource(R.drawable.ic_close)
            background = null
            contentDescription = getString(R.string.cancel)
            setOnClickListener { finish() }
        }, FrameLayout.LayoutParams(Ui.dp(this, 48), Ui.dp(this, 48),
            Gravity.TOP or Gravity.END).apply {
            topMargin = Ui.dp(this@KalkulatorActivity, 8)
            rightMargin = Ui.dp(this@KalkulatorActivity, 8)
        })
        setContentView(root)
    }
}
