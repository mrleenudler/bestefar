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
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton
import com.google.android.material.button.MaterialButtonToggleGroup
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * Optikk-kalkulator (musingsUI): omregning MOA ↔ MRAD ↔ cm ved valgt
 * avstand, pluss antall klikk for vanlige klikkverdier og egne
 * optikkprofiler. Fullskjerm med X i hjørnet. SMOA-valg påvirker
 * MOA-omregningen (2,908 vs 2,778 cm/100 m).
 */
class KalkulatorActivity : AppCompatActivity() {

    private var unit = "CM"   // CM | MOA | MRAD

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val store = Store.get(this)
        val root = FrameLayout(this)
        val content = Ui.col(this)

        content.addView(Ui.title(this, getString(R.string.kalkulator)))

        val distance = EditText(this).apply {
            hint = getString(R.string.kalk_distance)
            inputType = InputType.TYPE_CLASS_NUMBER
            setText("100")
        }
        content.addView(distance)

        val value = EditText(this).apply {
            hint = getString(R.string.kalk_value)
            inputType = InputType.TYPE_CLASS_NUMBER or
                InputType.TYPE_NUMBER_FLAG_DECIMAL or InputType.TYPE_NUMBER_FLAG_SIGNED
        }
        content.addView(value)

        val toggle = MaterialButtonToggleGroup(this).apply {
            isSingleSelection = true; isSelectionRequired = true
        }
        val unitButtons = listOf("CM" to "cm", "MOA" to "MOA", "MRAD" to "mrad").map { (k, l) ->
            MaterialButton(this, null,
                com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
                id = ViewGroup.generateViewId(); text = l; tag = k
            }
        }
        unitButtons.forEach { toggle.addView(it) }
        toggle.check(unitButtons[0].id)
        content.addView(toggle)

        val smoa = CheckBox(this).apply { text = getString(R.string.optic_smoa) }
        content.addView(smoa)

        val result = TextView(this).apply { textSize = 17f }
        content.addView(result, Ui.matchWrap(16, this))

        fun recompute() {
            val d = distance.text.toString().toIntOrNull() ?: 100
            val v = value.text.toString().replace(',', '.').toDoubleOrNull()
            if (v == null || d <= 0) {
                result.text = getString(R.string.kalk_help)
                return
            }
            val moaCm = if (smoa.isChecked) OpticProfile.SMOA_CM_PER_100M
                else OpticProfile.MOA_CM_PER_100M
            val cmPerMoa = moaCm * d / 100.0
            val cmPerMrad = 10.0 * d / 100.0
            val cm = when (unit) {
                "MOA" -> v * cmPerMoa
                "MRAD" -> v * cmPerMrad
                else -> v
            }
            val sb = StringBuilder()
            sb.appendLine("Ved $d m:")
            sb.appendLine("  %.1f cm".format(cm))
            sb.appendLine("  %.2f MOA".format(cm / cmPerMoa))
            sb.appendLine("  %.2f mrad".format(cm / cmPerMrad))
            sb.appendLine()
            sb.appendLine(getString(R.string.kalk_clicks))
            val clickOptions = mutableListOf<Pair<String, Double>>()
            OpticProfile.MOA_STEPS.forEach { s ->
                clickOptions.add("${OpticProfile.moaLabel(s)} MOA" to s * moaCm)
            }
            OpticProfile.MRAD_STEPS.forEach { s ->
                clickOptions.add("%.2f mrad".format(s).replace('.', ',') to s * 10.0)
            }
            store.optics().forEach { o ->
                clickOptions.add(o.displayName.ifBlank { o.brandModel } to o.clickCmPer100)
            }
            clickOptions.forEach { (label, cm100) ->
                val clicks = cm / (cm100 * d / 100.0)
                sb.appendLine("  $label: ${abs(clicks).roundToInt()} klikk " +
                    "(%.1f)".format(abs(clicks)))
            }
            result.text = sb.toString()
        }

        val watcher = object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
            override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
            override fun afterTextChanged(s: Editable?) = recompute()
        }
        distance.addTextChangedListener(watcher)
        value.addTextChangedListener(watcher)
        smoa.setOnCheckedChangeListener { _, _ -> recompute() }
        toggle.addOnButtonCheckedListener { _, id, checked ->
            if (checked) {
                unit = unitButtons.first { it.id == id }.tag as String
                recompute()
            }
        }
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
