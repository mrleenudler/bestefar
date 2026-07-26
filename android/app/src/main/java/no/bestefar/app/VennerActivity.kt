package no.bestefar.app

import android.os.Bundle
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

/**
 * Venner (musingsUI runde 4). FRONT-END-SKJELETT — ekte venne-/lagdata og
 * -deling krever konto + backend (se backend_spec.md). Bygget her: legg-til-
 * flyt, delingsvalg (visningsnavn perma-delt), lag med venner gruppert under,
 * og gråing av lista når deling er deaktivert.
 */
class VennerActivity : AppCompatActivity() {

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

    private val active get() = store.friendShareActive

    private fun rebuild() {
        content.removeAllViews()
        content.addView(Ui.title(this, getString(R.string.menu_friends)))

        content.addView(MaterialButton(this).apply {
            text = getString(R.string.friends_add)
            layoutParams = Ui.matchWrap(4, this@VennerActivity)
            setOnClickListener { onAddFriend() }
        })
        content.addView(CheckBox(this).apply {
            text = getString(R.string.profile_findable)
            isChecked = store.findable
            setOnCheckedChangeListener { _, on -> store.findable = on }
        })

        // Lag med tilhørende venner, deretter øvrige venner (musingsUI runde 4)
        val teams = store.teams().sortedWith(
            compareBy({ it.sortOrder }, { -it.memberCount }))
        val friends = store.friends()
        val greyed = !active

        teams.forEachIndexed { i, team ->
            val row = Ui.row(this)
            row.addView(TextView(this).apply {
                text = "▸ ${team.name}"; textSize = 17f; alpha = if (greyed) 0.4f else 1f
                layoutParams = LinearLayout.LayoutParams(0,
                    ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            })
            // Flytt opp/ned (musingsUI runde 4)
            row.addView(MaterialButton(this, null,
                com.google.android.material.R.attr.borderlessButtonStyle).apply {
                text = "▲"; setOnClickListener { moveTeam(team, -1) }
            })
            row.addView(MaterialButton(this, null,
                com.google.android.material.R.attr.borderlessButtonStyle).apply {
                text = "▼"; setOnClickListener { moveTeam(team, +1) }
            })
            content.addView(row)
            friends.filter { team.id in it.teamIds }.sortedBy { it.shownName.lowercase() }
                .forEach { addFriendButton(it, indent = true, greyed = greyed) }
        }
        friends.filter { f -> teams.none { it.id in f.teamIds } }
            .sortedBy { it.shownName.lowercase() }
            .forEach { addFriendButton(it, indent = false, greyed = greyed) }

        if (friends.isEmpty()) content.addView(Ui.hint(this, getString(R.string.friends_empty)))
        if (!store.findable) content.addView(Ui.hint(this,
            getString(R.string.friends_not_findable)))

        // Egne delingsvalg (visningsnavn perma-checket) — musingsUI runde 4
        content.addView(Ui.section(this, getString(R.string.research_i_share)))
        sharingCheckboxes()
    }

    private fun addFriendButton(f: Friend, indent: Boolean, greyed: Boolean) {
        val shots = f.shotsTotal?.let { " · $it skudd" } ?: ""
        content.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            text = f.shownName + shots
            alpha = if (greyed) 0.4f else 1f
            layoutParams = Ui.matchWrap(2, this@VennerActivity).apply {
                if (indent) marginStart = Ui.dp(this@VennerActivity, 24)
            }
            setOnClickListener {
                if (greyed) requireSharingDialog(R.string.friends_need_share_view)
                else friendDetail(f)
            }
        })
    }

    private fun moveTeam(team: Team, dir: Int) {
        val list = store.teams().sortedBy { it.sortOrder }.toMutableList()
        val idx = list.indexOfFirst { it.id == team.id }
        val j = idx + dir
        if (idx < 0 || j < 0 || j >= list.size) return
        val a = list[idx]; val b = list[j]
        val tmp = a.sortOrder; a.sortOrder = b.sortOrder; b.sortOrder = tmp
        store.saveTeams(list)
        rebuild()
    }

    private fun onAddFriend() {
        // Ingen info delt -> åpne delingsdialog med visningsnavn checket
        if (store.friendShare.isEmpty() || !active) {
            Dialogs.friendSharingDialog(this, store) { rebuild() }
            return
        }
        if (store.nickname.isBlank()) { requireDisplayName { addFriendMethods() }; return }
        addFriendMethods()
    }

    private fun requireDisplayName(after: () -> Unit) {
        val input = EditText(this).apply {
            hint = getString(R.string.profile_display_hint)
            filters = Ui.nameFilters()
        }
        Ui.capitalize(input)
        AlertDialog.Builder(this)
            .setMessage(R.string.friends_need_display_name)
            .setView(input)
            .setPositiveButton(R.string.save) { _, _ ->
                val n = input.text.toString().trim()
                if (n.isNotEmpty()) { store.nickname = n; after() }
            }
            .setNegativeButton(R.string.friends_dont_share, null)
            .show()
    }

    private fun addFriendMethods() {
        AlertDialog.Builder(this)
            .setTitle(R.string.friends_add)
            .setItems(arrayOf(getString(R.string.friends_search),
                getString(R.string.friends_enter_id),
                getString(R.string.friends_scan_qr),
                getString(R.string.friends_show_qr))) { _, _ ->
                Toast.makeText(this, R.string.friends_todo, Toast.LENGTH_LONG).show()
            }
            .show()
    }

    private fun requireSharingDialog(msgRes: Int) {
        AlertDialog.Builder(this)
            .setMessage(msgRes)
            .setPositiveButton(R.string.ok, null)
            .show()
    }

    private fun friendDetail(f: Friend) {
        content.removeAllViews()
        // Visningsnavn øverst med «endre visningsnavn»; original i parentes
        val header = Ui.row(this)
        val nameText = if (f.nickAlias.isNotBlank())
            "${f.nickAlias}  (${f.displayName})" else f.displayName
        header.addView(TextView(this).apply {
            text = nameText; textSize = 22f
            layoutParams = LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        })
        header.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = getString(R.string.friends_edit_alias)
            setOnClickListener { editAlias(f) }
        })
        content.addView(header)

        f.homeKommune?.let { content.addView(Ui.body(this, "Hjemkommune: $it")) }
        f.phone?.let { content.addView(Ui.body(this, "Telefon: $it  ☎  ✉")) }
        f.shotsTotal?.let { content.addView(Ui.body(this, "Øvelsesskudd totalt: $it")) }
        f.shotsSeason?.let { content.addView(Ui.body(this, "Denne sesongen: $it")) }
        content.addView(Ui.hint(this, getString(R.string.friends_data_note)))

        content.addView(MaterialButton(this).apply {
            text = getString(R.string.ok)
            layoutParams = Ui.matchWrap(16, this@VennerActivity)
            setOnClickListener { rebuild() }
        })
    }

    private fun editAlias(f: Friend) {
        val input = EditText(this).apply {
            hint = getString(R.string.friends_alias_hint)
            filters = Ui.nameFilters()
            setText(f.nickAlias)
        }
        AlertDialog.Builder(this)
            .setTitle(R.string.friends_edit_alias)
            .setView(input)
            .setPositiveButton(R.string.save) { _, _ ->
                f.nickAlias = input.text.toString().trim()
                store.updateFriend(f)
                friendDetail(f)
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun sharingCheckboxes() {
        // Visningsnavn perma-checket; øvrige unchecked default (musingsUI r4)
        content.addView(CheckBox(this).apply {
            text = getString(R.string.friend_share_name); isChecked = true; isEnabled = false
        })
        val items = listOf(
            "Skudd" to R.string.friend_share_shots,
            "Score" to R.string.friend_share_score,
            "Utvikling" to R.string.friend_share_trend,
            "Fellinger" to R.string.friend_share_kills,
            "Telefon" to R.string.friend_share_phone,
            "Lag" to R.string.friend_share_teams,
            "Hjemkommune" to R.string.friend_share_kommune)
        val current = store.friendShare
        val boxes = items.map { (key, res) ->
            CheckBox(this).apply { setText(res); isChecked = key in current; tag = key }
        }
        boxes.forEach { content.addView(it) }
        val save = MaterialButton(this).apply {
            text = getString(R.string.save)
            layoutParams = Ui.matchWrap(8, this@VennerActivity)
            setOnClickListener {
                store.friendShare = boxes.filter { it.isChecked }
                    .map { it.tag as String }.toSet() + "Navn"
                store.friendShareActive = true
                rebuild()
            }
        }
        content.addView(save)
        content.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = getString(R.string.sharing_deactivate)
            setOnClickListener {
                // Deaktiver: uncheck alle og lagre (musingsUI runde 4)
                store.friendShare = emptySet()
                store.friendShareActive = false
                rebuild()
            }
        })
    }
}
