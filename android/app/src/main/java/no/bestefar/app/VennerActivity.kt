package no.bestefar.app

import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.CheckBox
import android.widget.EditText
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton

/**
 * Venner (musingsUI runde 5). FRONT-END-SKJELETT (backend_spec.md). Lag som
 * innrykkede knapper med flytt- (opp/ned) og kollaps-pil; klikk på lag åpner
 * medlemssiden. Egne delingsvalg kan kollapses. «Lagre» går til hovedsiden.
 */
class VennerActivity : AppCompatActivity() {

    private lateinit var store: Store
    private lateinit var content: LinearLayout
    private val collapsedTeams = mutableSetOf<String>()
    private var shareCollapsed = false   // åpen som default (musingsUI runde 6)

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
            // Rebuild slik at varselet oppdateres forutsigbart (musingsUI runde 6)
            setOnCheckedChangeListener { _, on -> store.findable = on; rebuild() }
        })
        if (!store.findable) content.addView(Ui.hint(this,
            getString(R.string.findable_off_msg)))

        val greyed = !active
        val teams = store.teams().sortedBy { it.sortOrder }
        val friends = store.friends()
        val myTeamIds = teams.map { it.id }.toSet()

        teams.forEach { team ->
            val row = Ui.row(this)
            row.addView(MaterialButton(this, null,
                com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
                text = "▸ ${team.name}"
                alpha = if (greyed) 0.4f else 1f
                layoutParams = LinearLayout.LayoutParams(0,
                    ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
                setOnClickListener { if (!greyed) teamPage(team) }
            })
            // «<=» kollaps-pil, nest lengst til høyre (musingsUI runde 5)
            row.addView(MaterialButton(this, null,
                com.google.android.material.R.attr.borderlessButtonStyle).apply {
                text = if (team.id in collapsedTeams) "⇦" else "⇨"
                setOnClickListener {
                    if (team.id in collapsedTeams) collapsedTeams.remove(team.id)
                    else collapsedTeams.add(team.id)
                    rebuild()
                }
            })
            // Opp/ned-ikon lengst til høyre -> popup med flytt opp/ned/avbryt
            row.addView(ImageButton(this).apply {
                setImageResource(R.drawable.ic_up_down)
                background = null
                contentDescription = getString(R.string.move)
                setOnClickListener { moveTeamPopup(team) }
            })
            content.addView(row)

            if (team.id !in collapsedTeams) {
                // Egen bruker vises også i laget (musingsUI runde 6)
                content.addView(selfInTeamButton(greyed))
                // Venner i flere lag føres opp under hvert (samme runde)
                friends.filter { team.id in it.teamIds }.sortedBy { it.shownName.lowercase() }
                    .forEach { addFriendButton(it, indent = true, greyed = greyed) }
            }
        }
        // Kun venner du IKKE deler lag med, vises utenfor lagene (runde 6)
        friends.filter { f -> f.teamIds.none { it in myTeamIds } }
            .sortedBy { it.shownName.lowercase() }
            .forEach { addFriendButton(it, indent = false, greyed = greyed) }

        if (friends.isEmpty() && teams.isEmpty())
            content.addView(Ui.hint(this, getString(R.string.friends_empty)))

        sharingSection()
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

    private fun selfInTeamButton(greyed: Boolean) = MaterialButton(this, null,
        com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
        text = store.nickname.ifBlank { getString(R.string.team_you) } +
            " ${getString(R.string.team_you)}"
        alpha = if (greyed) 0.4f else 1f
        layoutParams = Ui.matchWrap(2, this@VennerActivity).apply {
            marginStart = Ui.dp(this@VennerActivity, 24)
        }
    }

    private fun moveTeamPopup(team: Team) {
        // «Avbryt» til høyre (positiv knapp) — musingsUI runde 6
        AlertDialog.Builder(this)
            .setItems(arrayOf(getString(R.string.move_up), getString(R.string.move_down))) {
                _, which -> if (which == 0) moveTeam(team, -1) else moveTeam(team, +1)
            }
            .setPositiveButton(R.string.cancel, null)
            .show()
    }

    private fun moveTeam(team: Team, dir: Int) {
        // Normaliser sortOrder til indeks foerst, slik at bytte faktisk endrer
        // rekkefoelgen (musingsUI runde 6-bug: like sortOrder ga ingen effekt).
        val list = store.teams().sortedBy { it.sortOrder }.toMutableList()
        list.forEachIndexed { i, t -> t.sortOrder = i }
        val idx = list.indexOfFirst { it.id == team.id }
        val j = idx + dir
        if (idx < 0 || j < 0 || j >= list.size) return
        val tmp = list[idx].sortOrder; list[idx].sortOrder = list[j].sortOrder
        list[j].sortOrder = tmp
        store.saveTeams(list); rebuild()
    }

    /** Lagside: laget som overskrift, medlemmer som knapper (musingsUI runde 5). */
    private fun teamPage(team: Team) {
        content.removeAllViews()
        content.addView(Ui.title(this, team.name))
        val members = store.friends().filter { team.id in it.teamIds }
        if (members.isEmpty()) content.addView(Ui.hint(this, getString(R.string.team_no_members)))
        members.sortedBy { it.shownName.lowercase() }.forEach { f ->
            content.addView(MaterialButton(this, null,
                com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
                text = f.shownName
                layoutParams = Ui.matchWrap(2, this@VennerActivity)
                setOnClickListener { friendDetail(f) }
            })
        }
        content.addView(MaterialButton(this).apply {
            text = getString(R.string.back)
            layoutParams = Ui.matchWrap(16, this@VennerActivity)
            setOnClickListener { rebuild() }
        })
    }

    private fun onAddFriend() {
        if (store.friendShare.isEmpty() || !active) {
            Dialogs.friendSharingDialog(this, store) { rebuild() }; return
        }
        if (store.nickname.isBlank()) { requireDisplayName { addFriendMethods() }; return }
        addFriendMethods()
    }

    private fun requireDisplayName(after: () -> Unit) {
        val input = EditText(this).apply {
            hint = getString(R.string.profile_display_hint); filters = Ui.nameFilters()
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
                Ui.toast(this, R.string.friends_todo)
            }
            .show()
    }

    private fun requireSharingDialog(msgRes: Int) {
        AlertDialog.Builder(this).setMessage(msgRes)
            .setPositiveButton(R.string.ok, null).show()
    }

    private fun friendDetail(f: Friend) {
        content.removeAllViews()
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
        content.addView(Ui.hint(this, getString(R.string.friends_data_note)))
        content.addView(MaterialButton(this).apply {
            text = getString(R.string.back)
            layoutParams = Ui.matchWrap(16, this@VennerActivity)
            setOnClickListener { rebuild() }
        })
    }

    private fun editAlias(f: Friend) {
        val input = EditText(this).apply {
            hint = getString(R.string.friends_alias_hint)
            filters = Ui.nameFilters(); setText(f.nickAlias)
        }
        AlertDialog.Builder(this)
            .setTitle(R.string.friends_edit_alias)
            .setView(input)
            .setPositiveButton(R.string.save) { _, _ ->
                f.nickAlias = input.text.toString().trim()
                store.updateFriend(f); friendDetail(f)
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun sharingSection() {
        // «Jeg vil dele:» med «<=» kollaps (musingsUI runde 5)
        val head = Ui.row(this)
        head.addView(Ui.section(this, getString(R.string.research_i_share)).apply {
            layoutParams = LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        })
        head.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            // Kraftigere piler (musingsUI runde 6)
            text = if (shareCollapsed) "▼" else "◀"
            textSize = 22f
            setTypeface(typeface, android.graphics.Typeface.BOLD)
            setOnClickListener { shareCollapsed = !shareCollapsed; rebuild() }
        })
        content.addView(head)
        if (shareCollapsed) return

        content.addView(CheckBox(this).apply {
            text = getString(R.string.friend_share_name); isChecked = true; isEnabled = false
        })
        // Rekkefølge/etiketter per musingsUI runde 5; Telefon nederst
        val items = listOf(
            "Skudd" to R.string.friend_share_shots,
            "Score" to R.string.friend_share_score,
            "Utvikling" to R.string.friend_share_trend,
            "Fellinger" to R.string.friend_share_kills,
            "Lag" to R.string.friend_share_teams,
            "Hjemkommune" to R.string.friend_share_kommune,
            "Telefon" to R.string.friend_share_phone)
        val current = store.friendShare
        val boxes = items.map { (key, res) ->
            CheckBox(this).apply { setText(res); isChecked = key in current; tag = key }
        }
        boxes.forEach { content.addView(it) }
        content.addView(MaterialButton(this).apply {
            text = getString(R.string.save)
            layoutParams = Ui.matchWrap(8, this@VennerActivity)
            setOnClickListener {
                store.friendShare = boxes.filter { it.isChecked }
                    .map { it.tag as String }.toSet() + "Navn"
                store.friendShareActive = true
                // Lagre lukker menyen og går til hovedsiden (musingsUI runde 5)
                MainActivity.returnToHome = true
                finish()
            }
        })
        content.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = getString(R.string.sharing_deactivate)
            setOnClickListener {
                store.friendShare = emptySet(); store.friendShareActive = false; rebuild()
            }
        })
    }
}
