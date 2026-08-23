package no.bestefar.app

import android.content.Intent
import android.os.Bundle
import android.widget.LinearLayout
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton

/**
 * Tar imot invitasjonslenken `https://bestefar-api.fly.dev/i/<token>` og
 * loeser tokenet inn til medlemskap via `POST /v1/teams/join`.
 *
 * ## Hvorfor lenken ikke naadde hit foer
 *
 * To ting manglet, og begge maatte til (issue #15). Manifestet erklaerte ikke
 * domenet, saa lenken gikk til nettleseren og videre til Play — der den ga
 * «item not found» fordi appen ikke er publisert (AAP-E8). Og selv om den hadde
 * naadd appen, fantes det ingen kaller for `join`: `Teams` kunne bare *sende*
 * invitasjoner.
 *
 * ## App Links verifiseres mot signaturen, og feiler stille
 *
 * `android:autoVerify` sjekker appens signeringssertifikat mot
 * `/.well-known/assetlinks.json`. **Et debug-bygg er signert med
 * debug-keystoren og verifiserer derfor ikke** — lenken aapner bare nettleseren,
 * uten feilmelding noe sted. Test paa release-APK, og sjekk status med
 * `adb shell pm get-app-links no.bestefar.app` (skal si `verified`).
 *
 * ## Innlogging
 *
 * `join` krever konto. Er brukeren ikke innlogget, holder vi tokenet og tilbyr
 * innlogging; [onResume] proever paa nytt naar hen kommer tilbake. Tokenet
 * lagres bevisst IKKE utenfor denne aktiviteten — en invitasjon som blir
 * liggende i preferansene og loeses inn ved en senere anledning, er en
 * overraskelse ingen ba om.
 */
class InviteActivity : AppCompatActivity() {

    private var token: String = ""
    private var ferdig = false
    private var henter = false
    private lateinit var content: LinearLayout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        content = Ui.col(this)
        val scroller = Ui.scroll(this, content)
        Ui.applyInsets(scroller)
        setContentView(scroller)

        lesToken(intent)
    }

    /**
     * `launchMode="singleTask"`: en ny lenke mens aktiviteten lever kommer hit,
     * ikke til [onCreate]. Uten dette ville andre invitasjon blitt loest inn med
     * FOERSTE token — riktig kvittering, feil lag.
     */
    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        ferdig = false
        lesToken(intent)
        if (token.isNotBlank() && !henter) join()
    }

    /** Siste ledd i stien er tokenet: `/i/<token>`. */
    private fun lesToken(i: Intent?) {
        token = i?.data?.lastPathSegment.orEmpty()
        if (token.isBlank()) visning(getString(R.string.invite_bad_link))
        else vis(getString(R.string.invite_working))
    }

    override fun onResume() {
        super.onResume()
        if (token.isNotBlank() && !ferdig && !henter) join()
    }

    private fun join() {
        henter = true
        Teams.join(this, token) { u ->
            henter = false
            when (u) {
                is Teams.Utfall.Ok -> {
                    ferdig = true
                    // Laget legges i bufferet med det samme, saa laglista viser
                    // det ogsaa foer neste henting.
                    val kjent = Store.get(this).teams()
                    if (kjent.none { it.id == u.verdi.id })
                        Store.get(this).saveTeams(kjent + u.verdi)
                    visTeam(u.verdi)
                }
                is Teams.Utfall.IkkeInnlogget -> visLogginn()
                is Teams.Utfall.Feil -> when {
                    // Serveren skiller ikke ukjent fra utloept token; begge er
                    // «denne invitasjonen kan ikke brukes».
                    u.code == 404 -> visning(getString(R.string.invite_invalid))
                    u.retryable -> visning(getString(R.string.invite_offline),
                        prov = true)
                    else -> visning(getString(R.string.invite_rejected, u.code),
                        prov = true)
                }
            }
        }
    }

    // ---------- Visninger ----------

    private fun vis(tekst: String) {
        content.removeAllViews()
        content.addView(Ui.title(this, getString(R.string.invite_title)))
        content.addView(Ui.body(this, tekst))
    }

    private fun visning(tekst: String, prov: Boolean = false) {
        vis(tekst)
        if (prov) content.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            text = getString(R.string.team_retry)
            layoutParams = Ui.matchWrap(12, this@InviteActivity)
            setOnClickListener { vis(getString(R.string.invite_working)); join() }
        })
        lukkeknapp()
    }

    private fun visLogginn() {
        vis(getString(R.string.invite_needs_account))
        content.addView(MaterialButton(this).apply {
            text = getString(R.string.login_title)
            layoutParams = Ui.matchWrap(12, this@InviteActivity)
            // Ingen resultatkobling noedvendig: onResume proever igjen naar
            // brukeren kommer tilbake hit, uansett hvordan hen kom tilbake.
            setOnClickListener {
                startActivity(Intent(this@InviteActivity, LoggInnActivity::class.java))
            }
        })
        lukkeknapp()
    }

    private fun visTeam(t: Team) {
        vis(getString(R.string.invite_joined, t.name))
        content.addView(MaterialButton(this).apply {
            text = getString(R.string.invite_open_team)
            layoutParams = Ui.matchWrap(12, this@InviteActivity)
            setOnClickListener {
                startActivity(Intent(this@InviteActivity, TeamPageActivity::class.java)
                    .putExtra(TeamPageActivity.EXTRA_TEAM_ID, t.id))
                finish()
            }
        })
        lukkeknapp()
    }

    /**
     * Aktiviteten kan vaere startet fra en e-post, altsaa uten at appen var
     * aapen. `finish()` alene ville da latt brukeren staa igjen i e-postappen,
     * saa vi sender hen til hovedskjermen.
     */
    private fun lukkeknapp() {
        content.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = getString(R.string.close)
            layoutParams = Ui.matchWrap(4, this@InviteActivity)
            setOnClickListener {
                startActivity(Intent(this@InviteActivity, MainActivity::class.java)
                    .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP))
                finish()
            }
        })
    }
}
