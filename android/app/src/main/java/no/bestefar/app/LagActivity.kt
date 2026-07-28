package no.bestefar.app

import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.EditText
import android.widget.LinearLayout
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton

/**
 * Jaktlag og skytterlag — fullsides meny (musingsUI runde 5). FRONT-END-SKJELETT:
 * ekte medlemskap/roller/invitasjoner krever backend (backend_spec.md). «Mine
 * lag» som innrykkede knapper; klikk velger lag → Opprett-knappen blir Rediger.
 */
class LagActivity : AppCompatActivity() {

    private lateinit var store: Store
    private lateinit var content: LinearLayout
    private var selected: Team? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = Store.get(this)
        content = Ui.col(this)
        val scroller = Ui.scroll(this, content)
        Ui.applyInsets(scroller)
        setContentView(scroller)
        rebuild()
    }

    private fun rebuild() {
        content.removeAllViews()
        content.addView(Ui.title(this, getString(R.string.profile_add_team)))

        // Opprett / Rediger (avhengig av valgt lag)
        content.addView(MaterialButton(this).apply {
            text = getString(if (selected == null) R.string.team_create_plain
                else R.string.team_edit)
            layoutParams = Ui.matchWrap(4, this@LagActivity)
            setOnClickListener {
                val sel = selected
                if (sel == null) createTeamRoleDialog() else editTeamDialog(sel)
            }
        })

        content.addView(Ui.section(this, getString(R.string.team_mine)))
        val teams = store.teams()
        if (teams.isEmpty()) content.addView(Ui.hint(this, getString(R.string.team_none)))
        teams.forEach { t ->
            content.addView(MaterialButton(this, null,
                if (t.id == selected?.id)
                    com.google.android.material.R.attr.materialButtonStyle
                else com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
                text = t.name
                // Innrykk ift. Opprett/Rediger (avsnitt-stil, musingsUI runde 5)
                layoutParams = Ui.matchWrap(4, this@LagActivity).apply {
                    marginStart = Ui.dp(this@LagActivity, 24)
                }
                setOnClickListener {
                    selected = if (selected?.id == t.id) null else t
                    rebuild()
                }
            })
        }
        content.addView(Ui.hint(this, getString(R.string.team_backend_wait)))
    }

    private fun createTeamRoleDialog() {
        // Roller som knapper i samme stil (musingsUI runde 5); ingen invite-toast
        val col = Ui.col(this, 16)
        val dialog = AlertDialog.Builder(this)
            .setTitle(R.string.team_create_plain)
            .setView(col)
            .setNegativeButton(R.string.cancel, null)
            .create()
        listOf(R.string.team_role_leader, R.string.team_role_for_leader,
            R.string.team_role_ask_leader).forEach { res ->
            col.addView(MaterialButton(this, null,
                com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
                setText(res)
                layoutParams = Ui.matchWrap(4, this@LagActivity)
                setOnClickListener { dialog.dismiss(); nameTeamDialog() }
            })
        }
        dialog.show()
    }

    private fun nameTeamDialog() {
        val input = EditText(this).apply { hint = getString(R.string.team_name_hint) }
        Ui.capitalize(input)
        AlertDialog.Builder(this)
            .setTitle(R.string.team_create_plain)
            .setView(input)
            .setPositiveButton(R.string.save) { _, _ ->
                val n = input.text.toString().trim()
                if (n.isNotEmpty()) { store.addTeam(Team(Store.newId(), n)); rebuild() }
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun editTeamDialog(t: Team) {
        // Sole member -> slett uten varsel; flere -> forlat; sole leader -> valg.
        if (t.memberCount <= 1) {
            store.saveTeams(store.teams().filter { it.id != t.id })
            selected = null
            Ui.toast(this, R.string.team_deleted)
            rebuild()
            return
        }
        AlertDialog.Builder(this)
            .setTitle(t.name)
            .setItems(arrayOf(getString(R.string.team_leave),
                getString(R.string.team_dissolve),
                getString(R.string.team_transfer))) { _, which ->
                when (which) {
                    0, 1 -> {   // forlat / oppløs -> lokalt fjernes laget
                        store.saveTeams(store.teams().filter { it.id != t.id })
                        selected = null; rebuild()
                    }
                    2 -> Ui.toast(this, R.string.team_backend_wait)
                }
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }
}
