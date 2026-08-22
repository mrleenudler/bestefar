package no.bestefar.app

import android.os.Bundle
import android.widget.EditText
import android.widget.LinearLayout
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton

/**
 * Jaktlag og skytterlag — fullsides meny (musingsUI runde 5).
 *
 * **Fra v0.32 er lagene serverens** (`/v1/teams`, backend_spec.md §4). Fram til
 * da ble de opprettet lokalt med en UUID serveren aldri hadde sett, og ingen av
 * backendens 22 lag-ruter kunne kalles (AAP-U29). Lokale lag fra den tiden
 * migreres ikke — eierbeslutning 2026-08-22.
 *
 * ## Tre tilstander som ikke skal se like ut
 *
 * Skjermen skiller eksplisitt mellom «du har ingen lag», «vi fikk ikke kontakt»
 * og «du er ikke logget inn». Den mellomste er den viktige: en feilet henting
 * som tegnes som en tom liste forteller brukeren at lagene er borte.
 *
 * Ved feil vises **sist kjente liste fra bufferet** med en tydelig merknad om at
 * den ikke er oppdatert — appen er offline-foerst, og en gammel liste er mer
 * verdt enn ingen, saa lenge brukeren vet at den er gammel.
 */
class LagActivity : AppCompatActivity() {

    private lateinit var store: Store
    private lateinit var content: LinearLayout

    /** Siste utfall fra serveren. `null` = ikke forsoekt ennaa i denne oekten. */
    private var utfall: Teams.Utfall<List<Team>>? = null
    private var henter = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = Store.get(this)
        content = Ui.col(this)
        val scroller = Ui.scroll(this, content)
        Ui.applyInsets(scroller)
        setContentView(scroller)
        rebuild()
    }

    override fun onResume() { super.onResume(); refresh() }

    // ---------- Henting ----------

    private fun refresh() {
        if (henter) return
        henter = true
        rebuild()
        Teams.list(this) { u ->
            henter = false
            utfall = u
            // Bare et VELLYKKET svar faar roere bufferet. En feilet henting skal
            // ikke kunne toemme den lokale lista - det var nettopp slik en
            // manglende forbindelse ville sett ut som «lagene er slettet».
            if (u is Teams.Utfall.Ok) store.saveTeams(flettSortering(u.verdi))
            rebuild()
        }
    }

    /**
     * Serveren kjenner ikke [Team.sortOrder] — den er brukerens egen rekkefoelge
     * i venneskjermen. Behold den fra bufferet, og legg nye lag bakerst.
     */
    private fun flettSortering(fraServer: List<Team>): List<Team> {
        val gammel = store.teams().associateBy { it.id }
        var neste = (gammel.values.maxOfOrNull { it.sortOrder } ?: -1) + 1
        return fraServer.map { t ->
            val kjent = gammel[t.id]
            if (kjent != null) t.also { it.sortOrder = kjent.sortOrder }
            else t.also { it.sortOrder = neste++ }
        }
    }

    // ---------- Visning ----------

    private fun rebuild() {
        content.removeAllViews()
        content.addView(Ui.title(this, getString(R.string.profile_add_team)))

        val u = utfall
        if (u is Teams.Utfall.IkkeInnlogget) {
            content.addView(Ui.hint(this, getString(R.string.team_needs_account)))
            return
        }

        content.addView(MaterialButton(this).apply {
            text = getString(R.string.team_create_plain)
            layoutParams = Ui.matchWrap(4, this@LagActivity)
            setOnClickListener { createTeamRoleDialog() }
        })

        content.addView(Ui.section(this, getString(R.string.team_mine)))

        val bufret = store.teams().sortedBy { it.sortOrder }

        when {
            henter && bufret.isEmpty() ->
                content.addView(Ui.hint(this, getString(R.string.team_loading)))

            // Serveren svarte. Da ER en tom liste sannheten.
            u is Teams.Utfall.Ok ->
                if (u.verdi.isEmpty())
                    content.addView(Ui.hint(this, getString(R.string.team_none)))
                else bufret.forEach { addTeamButton(it) }

            u is Teams.Utfall.Feil -> {
                bufret.forEach { addTeamButton(it) }
                content.addView(Ui.hint(this, when {
                    // Et avvist kall er noe annet enn et kall som ikke kom fram.
                    !u.retryable && u.code != 0 ->
                        getString(R.string.team_load_rejected, u.code)
                    bufret.isEmpty() -> getString(R.string.team_load_failed_empty)
                    else -> getString(R.string.team_load_failed_cached)
                }))
                content.addView(MaterialButton(this, null,
                    com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
                    text = getString(R.string.team_retry)
                    layoutParams = Ui.matchWrap(8, this@LagActivity)
                    setOnClickListener { refresh() }
                })
            }

            // Ikke forsoekt ennaa (foerste tegning foer svaret er inne).
            else -> bufret.forEach { addTeamButton(it) }
        }
    }

    private fun addTeamButton(t: Team) {
        content.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            text = t.name
            // Innrykk (avsnitt-stil); klikk åpner laget (musingsUI runde 6)
            layoutParams = Ui.matchWrap(4, this@LagActivity).apply {
                marginStart = Ui.dp(this@LagActivity, 24)
            }
            setOnClickListener {
                startActivity(android.content.Intent(this@LagActivity,
                    TeamPageActivity::class.java)
                    .putExtra(TeamPageActivity.EXTRA_TEAM_ID, t.id))
            }
        })
    }

    // ---------- Oppretting ----------

    /**
     * §4-rollene. De to foerste oppretter et lag og skiller seg bare paa
     * `i_am_leader`; den tredje oppretter **ingenting** — aa be en leder
     * opprette laget er en invitasjon til en annen person, og den ruten er ikke
     * koblet ennaa. Fram til v0.31 laget alle tre et lag lokalt, ogsaa den
     * siste, som dermed gjorde det motsatte av det knappen sa.
     */
    private fun createTeamRoleDialog() {
        val col = Ui.col(this, 16)
        val dialog = AlertDialog.Builder(this)
            .setTitle(R.string.team_create_plain)
            .setView(col)
            .setNegativeButton(R.string.cancel, null)
            .create()
        listOf(
            R.string.team_role_leader to true,
            R.string.team_role_for_leader to false,
        ).forEach { (res, erLeder) ->
            col.addView(MaterialButton(this, null,
                com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
                setText(res)
                layoutParams = Ui.matchWrap(4, this@LagActivity)
                setOnClickListener { dialog.dismiss(); kindDialog(erLeder) }
            })
        }
        col.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            setText(R.string.team_role_ask_leader)
            layoutParams = Ui.matchWrap(4, this@LagActivity)
            setOnClickListener {
                dialog.dismiss()
                Ui.toast(this@LagActivity, R.string.team_ask_leader_todo)
            }
        })
        dialog.show()
    }

    /** `kind` er paakrevd av `POST /v1/teams` og har ingen fornuftig default. */
    private fun kindDialog(erLeder: Boolean) {
        AlertDialog.Builder(this)
            .setTitle(R.string.team_kind_title)
            .setItems(arrayOf(getString(R.string.team_kind_jakt),
                getString(R.string.team_kind_skytter))) { _, which ->
                nameTeamDialog(erLeder,
                    if (which == 0) Team.KIND_JAKT else Team.KIND_SKYTTER)
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun nameTeamDialog(erLeder: Boolean, kind: String) {
        val input = EditText(this).apply { hint = getString(R.string.team_name_hint) }
        Ui.capitalize(input)
        AlertDialog.Builder(this)
            .setTitle(R.string.team_create_plain)
            .setView(input)
            .setPositiveButton(R.string.save) { _, _ ->
                val n = input.text.toString().trim()
                if (n.isNotEmpty()) opprett(n, kind, erLeder)
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun opprett(navn: String, kind: String, erLeder: Boolean) {
        Teams.create(this, navn, kind, erLeder) { u ->
            when (u) {
                is Teams.Utfall.Ok -> {
                    Ui.toast(this, getString(R.string.team_created, u.verdi.name))
                    // Hent lista paa nytt framfor aa stole paa at det ene svaret
                    // er hele sannheten - da er skjermen riktig ogsaa om noe ble
                    // opprettet fra en annen enhet i mellomtiden.
                    refresh()
                }
                is Teams.Utfall.IkkeInnlogget ->
                    Ui.toast(this, R.string.team_needs_account)
                is Teams.Utfall.Feil ->
                    if (u.retryable) Ui.toast(this, R.string.team_create_failed)
                    else Ui.toast(this, getString(R.string.team_create_rejected, u.code))
            }
        }
    }
}
