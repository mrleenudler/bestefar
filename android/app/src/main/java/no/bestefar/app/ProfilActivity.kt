package no.bestefar.app

import android.os.Bundle
import android.text.Editable
import android.text.InputType
import android.text.TextWatcher
import android.view.Gravity
import android.view.ViewGroup
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton
import java.time.LocalDate

/**
 * Min profil (musingsUI runde 4): visningsnavn, fødselsår (2–120 år), «Mitt
 * jaktmål», tema-veksler øverst til høyre, fortløpende lagring. Våpenkartotek,
 * flytt og sletting ligger under «Avanserte innstillinger». Deling mot venner
 * skjer i vennelisten, ikke her.
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

    private fun watcher(onText: (String) -> Unit) = object : TextWatcher {
        override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
        override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
        override fun afterTextChanged(s: Editable?) = onText(s?.toString() ?: "")
    }

    private fun rebuild() {
        content.removeAllViews()

        // Tittel + tema-veksler øverst til høyre (musingsUI runde 4)
        val header = Ui.row(this)
        header.addView(Ui.title(this, getString(R.string.profile_title)).apply {
            layoutParams = LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        })
        header.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = getString(R.string.theme_button)
            setOnClickListener { themeDialog() }
        })
        content.addView(header)

        // Visningsnavn
        val nick = EditText(this).apply { hint = getString(R.string.profile_display_hint) }
        Ui.capitalize(nick)
        nick.setText(store.nickname)
        nick.addTextChangedListener(watcher { store.nickname = it.trim() })
        content.addView(nick)

        // Fødselsår (2–120 år)
        val birthRow = Ui.row(this)
        birthRow.addView(TextView(this).apply {
            text = getString(R.string.profile_birth_label); textSize = 16f
        })
        val birth = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER
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

        content.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            text = getString(R.string.profile_add_team)
            layoutParams = Ui.matchWrap(4, this@ProfilActivity)
            setOnClickListener { teamDialog() }
        })
        val teams = store.teams()
        if (teams.isNotEmpty()) {
            content.addView(Ui.hint(this, teams.joinToString(" · ") { it.name }))
        }

        content.addView(CheckBox(this).apply {
            text = getString(R.string.profile_findable)
            isChecked = store.findable
            setOnCheckedChangeListener { _, on -> store.findable = on }
        })

        // Mitt jaktmål (musingsUI runde 4)
        content.addView(Ui.section(this, getString(R.string.jaktmaal_title)))
        val goalRow = Ui.row(this)
        goalRow.addView(TextView(this).apply {
            text = getString(R.string.jaktmaal_current, Dialogs.rateLabel(store.rateLimit))
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

        // Avanserte innstillinger
        content.addView(Ui.section(this, getString(R.string.profile_advanced)))
        content.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            text = getString(R.string.profile_weapons_mine)
            layoutParams = Ui.matchWrap(4, this@ProfilActivity)
            setOnClickListener { weaponsDialog() }
        })
        content.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            text = getString(R.string.profile_move)
            layoutParams = Ui.matchWrap(4, this@ProfilActivity)
            setOnClickListener {
                Toast.makeText(this@ProfilActivity, R.string.profile_move_todo,
                    Toast.LENGTH_SHORT).show()
            }
        })
        content.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            text = getString(R.string.profile_delete)
            layoutParams = Ui.matchWrap(4, this@ProfilActivity)
            setOnClickListener {
                AlertDialog.Builder(this@ProfilActivity)
                    .setMessage(R.string.profile_delete_confirm)
                    .setPositiveButton(R.string.profile_delete) { _, _ -> store.wipeAll(); finish() }
                    .setNegativeButton(R.string.cancel, null)
                    .show()
            }
        })
    }

    private fun themeDialog() {
        val modes = listOf("light" to getString(R.string.theme_light),
            "dark" to getString(R.string.theme_dark),
            "system" to getString(R.string.theme_system))
        AlertDialog.Builder(this)
            .setTitle(R.string.theme_button)
            .setItems(modes.map { it.second }.toTypedArray()) { _, i ->
                store.themeMode = modes[i].first
                recreate()
            }
            .show()
    }

    private fun teamDialog() {
        // FRONT-END-SKJELETT: nærliggende lag (≤20 innen 50 km) krever backend
        // (se backend_spec.md). Her: opprett lokalt lag + rolleflyt-skjelett.
        val root = Ui.col(this, 16)
        root.addView(MaterialButton(this).apply {
            text = getString(R.string.team_create)
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { gravity = Gravity.CENTER_HORIZONTAL }
            setOnClickListener { createTeamRoleDialog() }
        })
        root.addView(Ui.hint(this, getString(R.string.team_nearby_todo)))
        store.teams().forEach { t ->
            root.addView(Ui.body(this, "• ${t.name}"))
        }
        AlertDialog.Builder(this)
            .setTitle(R.string.profile_add_team)
            .setView(androidx.core.widget.NestedScrollView(this).apply { addView(root) })
            .setNegativeButton(R.string.close, null)
            .show()
    }

    private fun createTeamRoleDialog() {
        val roles = arrayOf(getString(R.string.team_role_leader),
            getString(R.string.team_role_for_leader),
            getString(R.string.team_role_ask_leader))
        AlertDialog.Builder(this)
            .setTitle(R.string.team_create)
            .setItems(roles) { _, _ ->
                val input = EditText(this).apply { hint = getString(R.string.team_name_hint) }
                Ui.capitalize(input)
                AlertDialog.Builder(this)
                    .setTitle(R.string.team_create)
                    .setView(input)
                    .setPositiveButton(R.string.save) { _, _ ->
                        val n = input.text.toString().trim()
                        if (n.isNotEmpty()) {
                            store.addTeam(Team(Store.newId(), n))
                            Toast.makeText(this, R.string.team_invite_todo,
                                Toast.LENGTH_LONG).show()
                            rebuild()
                        }
                    }
                    .setNegativeButton(R.string.cancel, null)
                    .show()
            }
            .show()
    }

    private fun weaponsDialog() {
        val root = Ui.col(this, 16)
        val dialog = AlertDialog.Builder(this)
            .setTitle(R.string.profile_weapons_mine)
            .setView(androidx.core.widget.NestedScrollView(this).apply { addView(root) })
            .setNegativeButton(R.string.close, null)
            .create()
        fun fill() {
            root.removeAllViews()
            store.weapons().forEach { w ->
                val row = Ui.row(this)
                row.addView(TextView(this).apply {
                    text = w.shownName; textSize = 16f
                    layoutParams = LinearLayout.LayoutParams(0,
                        ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
                })
                row.addView(MaterialButton(this, null,
                    com.google.android.material.R.attr.borderlessButtonStyle).apply {
                    text = getString(R.string.change)
                    setOnClickListener {
                        Dialogs.weaponEdit(this@ProfilActivity, store, w) { fill() }
                    }
                })
                root.addView(row)
            }
            root.addView(MaterialButton(this).apply {
                text = getString(R.string.weapon_add)
                layoutParams = LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT
                ).apply { gravity = Gravity.CENTER_HORIZONTAL }
                setOnClickListener {
                    Dialogs.weaponEdit(this@ProfilActivity, store, null) { fill() }
                }
            })
        }
        fill()
        dialog.show()
    }
}
