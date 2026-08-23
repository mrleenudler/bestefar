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
    /**
     * ScrollView-en rundt [content], opprettet ÉN gang.
     *
     * `Ui.scroll` gjoer et rent `addView(content)` uten aa loesne barnet fra en
     * tidligere forelder. Ble den kalt paa nytt for hver [rebuild], kastet den
     * andre tegningen `IllegalStateException: The specified child already has a
     * parent` — `root.removeAllViews()` loesner ScrollView-en fra `root`, men
     * ScrollView-en holder fortsatt `content`.
     *
     * Fella laa latent saa lenge skjermen bare ble tegnet én gang. v0.33 la inn
     * detalj-kallet, saa `rebuild()` kjoerer minst to ganger per aapning (én fra
     * `onCreate`, én fra svaret) — og da krasjet den hver gang.
     */
    private lateinit var scroller: android.widget.ScrollView

    /** Siste utfall fra `GET /v1/teams/{id}`. `null` = ikke forsoekt ennaa. */
    private var utfall: Teams.Utfall<Team>? = null
    private var henter = false

    /** Ett medlem i lista/karusellen, utledet av serverens `members[]`. */
    private data class Member(
        val name: String, val isSelf: Boolean, val isLeader: Boolean,
        /** Intern UUID — det ruteparametrene tar imot. Tom for ukjent. */
        val userId: String = "",
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = Store.get(this)
        // Bufret utgave gir navnet med det samme; medlemmene kommer fra kallet.
        team = store.teams().firstOrNull { it.id == intent.getStringExtra(EXTRA_TEAM_ID) }
        root = FrameLayout(this)
        Ui.applyInsets(root)
        content = Ui.col(this)
        scroller = Ui.scroll(this, content)
        setContentView(root)
        rebuild()
    }

    override fun onResume() { super.onResume(); refresh() }

    private fun refresh() {
        val id = intent.getStringExtra(EXTRA_TEAM_ID) ?: return
        if (henter) return
        henter = true
        Teams.details(this, id) { u ->
            henter = false
            utfall = u
            // Bare et vellykket svar faar erstatte laget. En feilet henting skal
            // ikke kunne gjoere et lag med medlemmer om til et uten.
            if (u is Teams.Utfall.Ok) {
                val sort = team?.sortOrder ?: 0
                team = u.verdi.also { it.sortOrder = sort }
            }
            rebuild()
        }
    }

    private fun selfName() = store.nickname.ifBlank { getString(R.string.me_label) }

    /**
     * Medlemmer i visningsrekkefoelge: lagleder(e) oeverst merket «(Lagleder)»,
     * ellers alfabetisk (musingsUI runde 7).
     *
     * **Returnerer `null` naar medlemslista ikke er hentet** — ikke en tom
     * liste. Et lag uten medlemmer og et lag vi ikke fikk svar om skal aldri
     * tegnes likt (`Team.members`).
     *
     * «Meg» kjennes igjen paa `public_id`, som er den eneste bruker-ID-en
     * klienten kjenner om seg selv. **Aldri paa visningsnavn** — det feiler
     * stille i akkurat de lagene der to personer heter det samme, og utfallet
     * ville vaert at feil person ble merket «(Du)».
     */
    private fun members(t: Team): List<Member>? {
        val fraServer = t.members ?: return null
        val minId = store.accountPublicId
        return fraServer
            .map { m ->
                Member(
                    name = if (t.erMeg(m, minId)) selfName() else m.displayName,
                    isSelf = t.erMeg(m, minId),
                    isLeader = m.isLeader,
                    userId = m.userId,
                )
            }
            .sortedWith(compareByDescending<Member> { it.isLeader }
                .thenBy { it.name.lowercase() })
    }

    private fun addMemberButton(m: Member, i: Int) {
        content.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            text = display(m)
            layoutParams = Ui.matchWrap(2, this@TeamPageActivity)
            setOnClickListener { memberCarousel(i) }
        })
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
            setOnClickListener { inviteDialog(t) }
        })

        val mem = members(t)
        if (mem == null) {
            // Ikke hentet. Si det - ikke tegn en tom liste.
            val u = utfall
            content.addView(Ui.hint(this, when {
                henter || u == null -> getString(R.string.team_loading)
                u is Teams.Utfall.IkkeInnlogget -> getString(R.string.team_needs_account)
                else -> getString(R.string.team_members_unknown)
            }))
            if (!henter && u is Teams.Utfall.Feil) {
                content.addView(MaterialButton(this, null,
                    com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
                    text = getString(R.string.team_retry)
                    layoutParams = Ui.matchWrap(8, this@TeamPageActivity)
                    setOnClickListener { refresh() }
                })
            }
        } else {
            mem.forEachIndexed { i, m -> addMemberButton(m, i) }
            // Serveren svarte, og laget har bare meg.
            if (mem.none { !it.isSelf })
                content.addView(Ui.hint(this, getString(R.string.team_members_none)))
        }
        // Plass til den faste knapperaden nederst
        content.addView(android.widget.Space(this), LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, Ui.dp(this, 72)))
        // Gjenbruk den ene ScrollView-en. Se feltet: en ny per tegning gir
        // «child already has a parent» fra og med andre gang.
        root.addView(scroller, ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT)

        // Nederst venstre: «Rediger lag» til lagleder, «Velg leder» naar laget
        // staar uten leder. Gatingen leser `my_role` (Team.amLeader), ikke
        // `has_leader` - fram til v0.33 behandlet den «laget HAR en leder» som
        // «jeg ER lederen», og ga dermed redigeringsmenyen til alle i et lag
        // med leder. `leaders[]` brukes bevisst ikke: den er interne user_id-er
        // og er ikke ment for gjenkjenning (backend/KONTRAKT.md paragraf 6).
        //
        // amLeader er null naar rollen ikke er hentet. Da vises INGEN av delene:
        // vi vet ikke hva brukeren har lov til, og aa gjette gir enten en meny
        // som ikke virker eller en knapp som mangler.
        val erLeder = t.amLeader
        val leftLabel = when {
            erLeder == true -> R.string.team_edit
            erLeder == false && !t.hasLeader -> R.string.team_choose_leader
            else -> null
        }
        if (leftLabel != null) root.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            text = getString(leftLabel)
            setOnClickListener { if (erLeder == true) editTeam(t) else chooseLeader(t) }
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
        // Uten hentet medlemsliste finnes det ingen karusell aa aapne.
        val mem = members(t) ?: return
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
                    1 -> removeMemberDialog(t)
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

    // ---------- Invitasjon (§4) ----------

    /**
     * Inviter med e-post eller telefonnummer. Krever **medlemskap**, ikke
     * lederskap (`teams.py:invite` kaller `_medlemskap`), saa knappen vises til
     * alle i laget.
     */
    private fun inviteDialog(t: Team) {
        val input = EditText(this).apply {
            hint = getString(R.string.team_invite_hint)
            inputType = android.text.InputType.TYPE_CLASS_TEXT or
                android.text.InputType.TYPE_TEXT_VARIATION_EMAIL_ADDRESS
        }
        AlertDialog.Builder(this)
            .setTitle(R.string.team_invite_members)
            .setView(input)
            .setPositiveButton(R.string.save) { _, _ ->
                val v = input.text.toString().trim()
                if (v.isNotEmpty()) sendInvite(t, v)
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun sendInvite(t: Team, maal: String) {
        Teams.invite(this, t.id, maal) { u ->
            when (u) {
                is Teams.Utfall.Ok -> {
                    val inv = u.verdi
                    // 201 betyr «lagret», ikke «kom fram». Er den ikke levert,
                    // skal brukeren se det som noe hen maa gjoere noe med - og
                    // faa lenken, som er hele grunnen til at serveren gir den.
                    if (inv.levert) {
                        Ui.toast(this, getString(R.string.team_invite_sent, inv.maal))
                    } else {
                        AlertDialog.Builder(this)
                            .setTitle(R.string.team_invite_members)
                            .setMessage(getString(R.string.team_invite_not_sent, inv.feil))
                            .setPositiveButton(R.string.team_invite_share) { _, _ ->
                                delLenke(inv.url)
                            }
                            .setNegativeButton(R.string.cancel, null)
                            .show()
                    }
                }
                is Teams.Utfall.IkkeInnlogget ->
                    Ui.toast(this, R.string.team_needs_account)
                is Teams.Utfall.Feil -> when {
                    // 422: serveren klarte ikke tolke adressen/nummeret.
                    u.code == 422 -> Ui.toast(this, R.string.team_invite_invalid)
                    u.retryable -> Ui.toast(this, R.string.team_invite_failed)
                    else -> Ui.toast(this,
                        getString(R.string.team_invite_rejected, u.code))
                }
            }
        }
    }

    private fun delLenke(url: String) {
        if (url.isEmpty()) return
        startActivity(android.content.Intent.createChooser(
            android.content.Intent(android.content.Intent.ACTION_SEND).apply {
                type = "text/plain"
                putExtra(android.content.Intent.EXTRA_TEXT, url)
            }, getString(R.string.team_invite_share_title)))
    }

    // ---------- Fjern medlem (§11) ----------

    /**
     * Krever lagleder (serveren svarer 403 ellers), og naas bare fra
     * redigeringsmenyen, som allerede er gatet paa `my_role`.
     *
     * Merk de to tomme tilfellene, som ikke er det samme: medlemslista er ikke
     * HENTET, eller den er hentet og inneholder bare meg.
     */
    private fun removeMemberDialog(t: Team) {
        val mem = members(t) ?: run {
            Ui.toast(this, R.string.team_remove_needs_list); return
        }
        val andre = mem.filter { !it.isSelf && it.userId.isNotEmpty() }
        if (andre.isEmpty()) { Ui.toast(this, R.string.team_remove_none); return }
        AlertDialog.Builder(this)
            .setTitle(R.string.team_remove_members)
            .setItems(andre.map { display(it) }.toTypedArray()) { _, i ->
                val m = andre[i]
                Ui.warningDialog(this)
                    .setTitle(R.string.team_remove_members)
                    .setMessage(getString(R.string.team_remove_confirm, m.name))
                    .setNegativeButton(R.string.cancel, null)
                    .setPositiveButton(R.string.team_remove_members) { _, _ ->
                        removeMember(t, m)
                    }
                    .show()
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun removeMember(t: Team, m: Member) {
        Teams.removeMember(this, t.id, m.userId) { u ->
            when (u) {
                is Teams.Utfall.Ok -> {
                    Ui.toast(this, getString(R.string.team_remove_done, m.name))
                    // Hent lista paa nytt framfor aa fjerne raden lokalt: da er
                    // skjermen riktig ogsaa om noe annet endret seg samtidig.
                    refresh()
                }
                is Teams.Utfall.IkkeInnlogget ->
                    Ui.toast(this, R.string.team_needs_account)
                is Teams.Utfall.Feil ->
                    if (u.retryable) Ui.toast(this, R.string.team_remove_failed)
                    else Ui.toast(this, getString(R.string.team_remove_rejected, u.code))
            }
        }
    }

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
        // Uten medlemsliste finnes det ingen kandidater aa stemme paa.
        val kandidater = members(t) ?: run {
            Ui.toast(this, R.string.team_members_unknown); return
        }
        val names = kandidater.map { display(it) }.toTypedArray()
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
