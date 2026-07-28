package no.bestefar.app

import android.content.Intent
import android.graphics.Typeface
import android.os.Bundle
import android.text.Editable
import android.text.InputType
import android.text.TextWatcher
import android.view.ViewGroup
import android.widget.CheckBox
import android.widget.EditText
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton
import java.time.LocalDate

/**
 * Min profil (musingsUI runde 5): visningsnavn (latinske tegn), fødselsår
 * (2–120), lag-liste, «Mitt jaktmål» (uthevet tall + (i)), tema-veksler øverst
 * til høyre, fortløpende lagring. Avanserte innstillinger er egen knapp.
 */
class ProfilActivity : AppCompatActivity() {

    private lateinit var store: Store
    private lateinit var content: LinearLayout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = Store.get(this)
        content = Ui.col(this)
        val scroller = Ui.scroll(this, content)
        Ui.applyInsets(scroller)
        setContentView(scroller)
        rebuild()
    }

    override fun onResume() { super.onResume(); rebuild() }

    private fun watcher(onText: (String) -> Unit) = object : TextWatcher {
        override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
        override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
        override fun afterTextChanged(s: Editable?) = onText(s?.toString() ?: "")
    }

    private fun themeLabel() = when (store.themeMode) {
        "dark" -> getString(R.string.theme_dark)
        "system" -> getString(R.string.theme_system)
        else -> getString(R.string.theme_light)
    }

    private fun rebuild() {
        content.removeAllViews()

        val header = Ui.row(this)
        header.addView(Ui.title(this, getString(R.string.profile_title)).apply {
            layoutParams = LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        })
        header.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = themeLabel()   // viser gjeldende modus (musingsUI runde 5)
            setOnClickListener { themeDialog() }
        })
        content.addView(header)

        val nick = EditText(this).apply {
            hint = getString(R.string.profile_display_hint)
            filters = Ui.nameFilters()
        }
        Ui.capitalize(nick)
        nick.setText(store.nickname)
        nick.addTextChangedListener(watcher { store.nickname = it.trim() })
        content.addView(nick)

        val birthRow = Ui.row(this)
        birthRow.addView(TextView(this).apply {
            text = getString(R.string.profile_birth_label); textSize = 16f
        })
        val birth = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER
            filters = arrayOf(android.text.InputFilter.LengthFilter(4), Ui.noLeadingZero())
            minWidth = Ui.dp(this@ProfilActivity, 90)
            setText(if (store.birthYear == 0) "" else store.birthYear.toString())
        }
        birth.addTextChangedListener(watcher { txt ->
            val y = txt.toIntOrNull() ?: return@watcher
            val now = LocalDate.now().year
            if (y in (now - 120)..(now - 2)) store.birthYear = y
        })
        birthRow.addView(birth)
        content.addView(birthRow)

        // Mine jaktlag og skytterlag (musingsUI runde 5)
        content.addView(Ui.section(this, getString(R.string.profile_teams_title)))
        store.teams().forEach { t ->
            content.addView(MaterialButton(this, null,
                com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
                text = t.name
                layoutParams = Ui.matchWrap(2, this@ProfilActivity)
                setOnClickListener { startActivity(Intent(this@ProfilActivity, LagActivity::class.java)) }
            })
        }
        content.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = getString(R.string.profile_add_more)
            setOnClickListener { startActivity(Intent(this@ProfilActivity, LagActivity::class.java)) }
        })

        content.addView(CheckBox(this).apply {
            text = getString(R.string.profile_findable)
            isChecked = store.findable
            setOnCheckedChangeListener { _, on -> store.findable = on }
        })

        // Mitt jaktmål: (i) høyrejustert på overskriftslinjen (musingsUI runde 5)
        val goalHeader = Ui.row(this)
        goalHeader.addView(Ui.section(this, getString(R.string.jaktmaal_title)).apply {
            layoutParams = LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        })
        goalHeader.addView(ImageButton(this).apply {
            setImageResource(R.drawable.ic_info)
            background = null
            contentDescription = getString(R.string.jaktmaal_why)
            setOnClickListener {
                AlertDialog.Builder(this@ProfilActivity)
                    .setMessage(R.string.jaktmaal_info)
                    .setPositiveButton(R.string.ok, null).show()
            }
        })
        content.addView(goalHeader)

        val goalRow = Ui.row(this)
        // Måltallet 2 pt større og bold (musingsUI runde 5)
        goalRow.addView(TextView(this).apply {
            text = getString(R.string.jaktmaal_prefix) + " "
            textSize = 16f
        })
        goalRow.addView(TextView(this).apply {
            text = Dialogs.rateLabel(store.rateLimit)
            textSize = 18f
            setTypeface(typeface, Typeface.BOLD)
        })
        goalRow.addView(TextView(this).apply {
            text = " " + getString(R.string.jaktmaal_suffix)
            textSize = 16f
            layoutParams = LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        })
        goalRow.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = getString(R.string.change)
            setOnClickListener { Dialogs.jaktmaalDialog(this@ProfilActivity, store) { rebuild() } }
        })
        content.addView(goalRow)

        // Avanserte innstillinger som egen knapp -> undermeny
        content.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            text = getString(R.string.profile_advanced)
            layoutParams = Ui.matchWrap(20, this@ProfilActivity)
            setOnClickListener { startActivity(Intent(this@ProfilActivity, AvansertActivity::class.java)) }
        })
    }

    private fun themeDialog() {
        val modes = listOf("light" to getString(R.string.theme_light),
            "dark" to getString(R.string.theme_dark),
            "system" to getString(R.string.theme_system))
        AlertDialog.Builder(this)
            .setTitle(R.string.theme_choose)   // «Velg visningsprofil»
            .setItems(modes.map { it.second }.toTypedArray()) { _, i ->
                store.themeMode = modes[i].first
                recreate()
            }
            .show()
    }
}
