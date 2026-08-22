package no.bestefar.app

import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton

/**
 * Jaktlag-/skytterlag-side (musingsUI runde 6/7). FRONT-END-SKJELETT: medlemskap,
 * lederskap, invitasjoner, avstemning og push-varsler krever backend
 * (backend_spec.md §4/§11). Her: navn, Inviter medlemmer, medlemsliste (leder
 * øverst, ellers alfabetisk), klikkbar medlems-karusell, Rediger lag / Velg
 * leder, Lukk. «Slett lag» håndterer forlat/oppløs/overfør (runde 7).
 */
class TeamPageActivity : AppCompatActivity() {

    companion object { const val EXTRA_TEAM_ID = "team_id" }

    private lateinit var store: Store
    private var team: Team? = null
    private lateinit var root: FrameLayout
    private lateinit var content: LinearLayout

    /** Ett medlem i lista/karusellen. */
    private data class Member(val name: String, val isSelf: Boolean, val isLeader: Boolean)

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

    private fun selfName() = store.nickname.ifBlank { getString(R.string.me_label) }

    /**
     * Medlemmer i visningsrekkefølge: lagleder(e) øverst merket «(Lagleder)»,
     * ellers alfabetisk (musingsUI runde 7). I skjelettet er egen bruker leder
     * når laget har leder.
     */
    private fun members(t: Team): List<Member> {
        val self = Member(selfName(), isSelf = true, isLeader = t.hasLeader)
        val friends = store.friends().filter { t.id in it.teamIds }
            .map { Member(it.shownName, isSelf = false, isLeader = false) }
        return if (t.hasLeader) {
            listOf(self) + friends.sortedBy { it.name.lowercase() }
        } else {
            (listOf(self) + friends).sortedBy { it.name.lowercase() }
        }
    }

    private fun display(m: Member): String {
        val you = if (m.isSelf) " ${getString(R.string.team_you)}" else ""
        val leader = if (m.isLeader) " ${getString(R.string.team_leader_tag)}" else ""
        return m.name + you + leader
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

        val mem = members(t)
        mem.forEachIndexed { i, m ->
            content.addView(MaterialButton(this, null,
                com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
                text = display(m)
                layoutParams = Ui.matchWrap(2, this@TeamPageActivity)
                setOnClickListener { memberCarousel(i) }
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

    // ---------- Medlems-karusell (musingsUI runde 7) ----------

    /**
     * Karusell over medlemmene med samme layout som ellers (piler + OK), men her
     * går pilene helt rundt (wraparound).
     */
    private fun memberCarousel(index: Int) {
        val t = team ?: return
        val mem = members(t)
        if (mem.isEmpty()) return
        val i = ((index % mem.size) + mem.size) % mem.size
        val m = mem[i]
        root.removeAllViews()
        val col = Ui.col(this)

        col.addView(Ui.title(this, display(m)))
        val friend = if (!m.isSelf)
            store.friends().firstOrNull { t.id in it.teamIds && it.shownName == m.name } else null
        when {
            m.isSelf -> col.addView(Ui.body(this, getString(R.string.team_you_detail)))
            friend != null -> {
                friend.homeKommune?.let { col.addView(Ui.body(this, "Hjemkommune: $it")) }
                friend.phone?.let { col.addView(Ui.body(this, "Telefon: $it  ☎  ✉")) }
                friend.shotsTotal?.let { col.addView(Ui.body(this, "Øvelsesskudd totalt: $it")) }
                col.addView(Ui.hint(this, getString(R.string.friends_data_note)))
            }
            else -> col.addView(Ui.hint(this, getString(R.string.friends_data_note)))
        }

        col.addView(android.widget.Space(this), LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, Ui.dp(this, 90)))
        root.addView(Ui.scroll(this, col), ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT)

        // Fast knapperad: piler går helt rundt, OK sentrert
        val bar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
        }
        val left = FrameLayout(this).apply {
            addView(bigArrow("‹") { memberCarousel(i - 1) })
        }
        bar.addView(left, LinearLayout.LayoutParams(0,
            ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        bar.addView(MaterialButton(this).apply {
            text = getString(R.string.ok)
            minWidth = Ui.dp(this@TeamPageActivity, 120)
            setOnClickListener { rebuild() }
        })
        val right = FrameLayout(this).apply {
            addView(bigArrow("›") { memberCarousel(i + 1) })
        }
        bar.addView(right, LinearLayout.LayoutParams(0,
            ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        root.addView(bar, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT,
            Gravity.BOTTOM).apply {
            bottomMargin = Ui.dp(this@TeamPageActivity, 24)
            leftMargin = Ui.dp(this@TeamPageActivity, 8)
            rightMargin = Ui.dp(this@TeamPageActivity, 8)
        })
    }

    /** Store, høye piler — samme stil som «Se registrerte skudd» (runde 6/7). */
    private fun bigArrow(glyph: String, onClick: () -> Unit) = MaterialButton(this, null,
        com.google.android.material.R.attr.borderlessButtonStyle).apply {
        text = glyph; textSize = 40f; scaleY = 1.7f
        setOnClickListener { onClick() }
    }

    // ---------- Rediger / slett ----------

    private fun editTeam(t: Team) {
        AlertDialog.Builder(this)
            .setTitle(R.string.team_edit)
            .setItems(arrayOf(getString(R.string.team_edit_name),
                getString(R.string.team_remove_members),
                getString(R.string.team_transfer),
                getString(R.string.team_delete))) { _, which ->
                when (which) {
                    0 -> renameTeam(t)
                    1 -> Ui.toast(this, R.string.team_backend_wait)   // varsler = backend
                    // Menyvalget heter «Overfoer lederskap» og skal derfor til
                    // overfoeringen, ikke til avstemningen. chooseLeader er
                    // «Velg leder» naar laget STAAR uten leder (linje 101) -
                    // to ulike §11-flyter som delte funksjon fram til v0.31.
                    2 -> offerLeadership(t)
                    3 -> deleteOrLeaveTeam(t)
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

    /**
     * Slett/forlat lag (musingsUI runde 7):
     *  - eneste medlem  -> slett laget uten varsel
     *  - flere medlemmer, ikke eneste leder -> forlat laget
     *  - eneste leder m/flere medlemmer -> oppløs for alle ELLER overfør + forlat
     */
    private fun deleteOrLeaveTeam(t: Team) {
        val others = store.friends().filter { t.id in it.teamIds }
        when {
            others.isEmpty() -> { removeTeamLocally(t); Ui.toast(this, R.string.team_deleted); finish() }
            t.hasLeader -> AlertDialog.Builder(this)
                .setTitle(t.name)
                .setItems(arrayOf(getString(R.string.team_dissolve),
                    getString(R.string.team_transfer))) { _, which ->
                    when (which) {
                        0 -> { removeTeamLocally(t); Ui.toast(this, R.string.team_deleted); finish() }
                        // Overfoering er IKKE en utmelding, og avslutter derfor
                        // ikke skjermen. Se offerLeadership.
                        1 -> offerLeadership(t)
                    }
                }
                .setNegativeButton(R.string.cancel, null)
                .show()
            else -> AlertDialog.Builder(this)
                .setMessage(getString(R.string.team_leave) + "?")
                .setPositiveButton(R.string.team_leave) { _, _ ->
                    removeTeamLocally(t); finish() }
                .setNegativeButton(R.string.cancel, null)
                .show()
        }
    }

    private fun removeTeamLocally(t: Team) =
        store.saveTeams(store.teams().filter { it.id != t.id })

    /**
     * Overfoer lederskap (`backend_spec.md` §11).
     *
     * **SLETTER INGENTING.** Semantikken er at lederskapet flyttes mens den
     * gamle lederen blir VAERENDE som vanlig medlem, og at byttet skjer foerst
     * naar den valgte BEKREFTER — «ingen skal vaakne opp som lagleder uten aa
     * ha sagt ja». Knappen starter altsaa en foresporsel; den fullfoerer ikke
     * en overfoering.
     *
     * Fram til v0.31 gjorde denne to ting den ikke skulle: den slettet laget
     * lokalt (ogsaa naar det ikke fantes andre medlemmer i det hele tatt) og
     * viste en kvittering. Brukeren mistet laget sitt uten at noen overfoering
     * skjedde noe sted.
     *
     * Kallet til `POST /v1/teams/{id}/leaders/{member_id}` er IKKE koblet til
     * enda, og kan ikke kobles foer laget finnes paa serveren: `Team.id` er en
     * lokal UUID fra `Store.newId()`, og klienten har aldri kalt
     * `POST /v1/teams`. Se `AAPNE_PUNKTER.md` AAP-U29.
     */
    private fun offerLeadership(t: Team) {
        val friends = store.friends().filter { t.id in it.teamIds }
        if (friends.isEmpty()) { Ui.toast(this, R.string.team_transfer_none); return }
        AlertDialog.Builder(this)
            .setTitle(R.string.team_transfer)
            .setItems(friends.map { it.shownName }.toTypedArray()) { _, _ ->
                Ui.toast(this, R.string.team_transfer_offline)
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    /** Velg ny leder: klikk et medlem, bekreft (musingsUI runde 6-skjelett). */
    private fun chooseLeader(t: Team) {
        val names = members(t).map { display(it) }.toTypedArray()
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
            .setNegativeButton(R.string.cancel, null)
            .show()
    }
}
