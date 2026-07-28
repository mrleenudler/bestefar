package no.bestefar.app

import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton

/**
 * Jaktlag-/skytterlag-side (musingsUI runde 6). FRONT-END-SKJELETT: medlemskap,
 * lederskap, invitasjoner, avstemning og push-varsler krever backend
 * (backend_spec.md §4/§11). Her: navn, Inviter medlemmer, medlemsliste (egen
 * bruker + venner i laget), Rediger lag / Velg leder, Lukk.
 */
class TeamPageActivity : AppCompatActivity() {

    companion object { const val EXTRA_TEAM_ID = "team_id" }

    private lateinit var store: Store
    private var team: Team? = null
    private lateinit var root: FrameLayout
    private lateinit var content: LinearLayout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = Store.get(this)
        team = store.teams().firstOrNull { it.id == intent.getStringExtra(EXTRA_TEAM_ID) }
        root = FrameLayout(this)
        Ui.applyInsets(root)
        content = Ui.col(this)
        setContentView(root)
        rebuild()
    }

    private fun members(t: Team): List<String> {
        // Egen bruker vises også (musingsUI runde 6)
        val self = store.nickname.ifBlank { getString(R.string.team_you) } +
            " ${getString(R.string.team_you)}"
        val friends = store.friends().filter { t.id in it.teamIds }.map { it.shownName }
        return listOf(self) + friends
    }

    private fun rebuild() {
        val t = team ?: run { finish(); return }
        root.removeAllViews()
        content.removeAllViews()

        content.addView(Ui.title(this, t.name))
        content.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            text = getString(R.string.team_invite_members)
            layoutParams = Ui.matchWrap(4, this@TeamPageActivity)
            setOnClickListener { Ui.toast(this@TeamPageActivity, R.string.team_backend_wait) }
        })

        members(t).forEach { name ->
            content.addView(MaterialButton(this, null,
                com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
                text = name
                layoutParams = Ui.matchWrap(2, this@TeamPageActivity)
                setOnClickListener { Ui.toast(this@TeamPageActivity, R.string.friends_data_note) }
            })
        }
        // Plass til den faste knapperaden nederst
        content.addView(android.widget.Space(this), LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, Ui.dp(this, 72)))
        root.addView(Ui.scroll(this, content), ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT)

        // Nederst venstre: Rediger lag (leder / eneste medlem) eller Velg leder
        val leftLabel = if (t.hasLeader) R.string.team_edit else R.string.team_choose_leader
        root.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            text = getString(leftLabel)
            setOnClickListener { if (t.hasLeader) editTeam(t) else chooseLeader(t) }
        }, FrameLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM or Gravity.START).apply {
            bottomMargin = Ui.dp(this@TeamPageActivity, 16)
            leftMargin = Ui.dp(this@TeamPageActivity, 16)
        })
        // Nederst høyre: Lukk
        root.addView(MaterialButton(this).apply {
            text = getString(R.string.close)
            setOnClickListener { finish() }
        }, FrameLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM or Gravity.END).apply {
            bottomMargin = Ui.dp(this@TeamPageActivity, 16)
            rightMargin = Ui.dp(this@TeamPageActivity, 16)
        })
    }

    private fun editTeam(t: Team) {
        AlertDialog.Builder(this)
            .setTitle(R.string.team_edit)
            .setItems(arrayOf(getString(R.string.team_edit_name),
                getString(R.string.team_remove_members),
                getString(R.string.team_transfer))) { _, which ->
                when (which) {
                    0 -> renameTeam(t)
                    1 -> Ui.toast(this, R.string.team_backend_wait)   // varsler = backend
                    2 -> chooseLeader(t)
                }
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun renameTeam(t: Team) {
        val input = EditText(this).apply { setText(t.name) }
        Ui.capitalize(input)
        AlertDialog.Builder(this)
            .setTitle(R.string.team_edit_name)
            .setView(input)
            .setPositiveButton(R.string.save) { _, _ ->
                val n = input.text.toString().trim()
                if (n.isNotEmpty()) {
                    store.saveTeams(store.teams().map {
                        if (it.id == t.id) it.copy(name = n) else it })
                    team = store.teams().firstOrNull { it.id == t.id }
                    // «Alle medlemmer informeres» -> backend push (skjelett)
                    Ui.toast(this, R.string.team_backend_wait)
                    rebuild()
                }
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    /** Velg ny leder: klikk et medlem, bekreft (musingsUI runde 6-skjelett). */
    private fun chooseLeader(t: Team) {
        val names = members(t).toTypedArray()
        AlertDialog.Builder(this)
            .setTitle(R.string.team_choose_leader)
            // Nedtellingstimer (7 dager) + push-avstemning krever backend (§11)
            .setItems(names) { _, i ->
                AlertDialog.Builder(this)
                    .setMessage(getString(R.string.team_confirm_leader, names[i]))
                    .setPositiveButton(R.string.team_choose_leader) { _, _ ->
                        Ui.toast(this, R.string.team_backend_wait)
                    }
                    .setNegativeButton(R.string.cancel, null)
                    .show()
            }
            .show()
    }
}
