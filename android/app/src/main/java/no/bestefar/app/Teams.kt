package no.bestefar.app

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject

/**
 * Nettverkslaget for lag (`/v1/teams`, backend_spec.md §4 og §11).
 *
 * **Dette laget fantes ikke foer v0.32.** Lagene var lokale, med UUID-er
 * serveren aldri hadde sett, og ingen av backendens ruter kunne kalles (AAP-U29).
 *
 * ## Utfallet er en type, ikke en tom liste
 *
 * Resten av appen er offline-foerst, og der er et mislykket kall normaltilstand:
 * [Messages.fetch] svarer med tom liste paa alt som kan gaa galt, fordi en
 * uleverte-meldinger-koe som ikke kunne hentes, ikke er noe brukeren skal
 * forstyrres med.
 *
 * **Lag er ikke slik.** «Laget ditt har ingen medlemmer» og «vi fikk ikke
 * kontakt» er to helt ulike beskjeder, og den foerste er skremmende naar den er
 * usann. Derfor returnerer alt her [Utfall], og kalleren maa ta stilling til
 * hvert tilfelle — en `emptyList()` kan ikke oppstaa ved en feil.
 */
object Teams {

    private const val TAG = "BestefarTeams"
    private const val PATH = "/v1/teams"

    /**
     * Resultatet av et lag-kall.
     *
     * [Feil] baerer [Api.Resp.retryable] videre fordi den avgjoer hva brukeren
     * skal se: et forbigaaende problem («proev igjen») er noe annet enn et
     * avvist kall («du er ikke medlem lenger»).
     */
    sealed class Utfall<out T> {
        data class Ok<T>(val verdi: T) : Utfall<T>()
        /** Kallet naadde ikke fram, eller serveren avviste det. IKKE «tomt». */
        data class Feil(val code: Int, val retryable: Boolean) : Utfall<Nothing>()
        /** Ingen konto. Lag krever innlogging; dette er ikke en feil. */
        object IkkeInnlogget : Utfall<Nothing>()
    }

    private fun <T> feil(r: Api.Resp): Utfall<T> {
        Log.d(TAG, "lag-kall feilet: code=${r.code} retryable=${r.retryable}")
        return Utfall.Feil(r.code, r.retryable)
    }

    // ---------- Henting ----------

    /**
     * `GET /v1/teams` — lagene jeg er medlem av. Listeformen har **ingen**
     * medlemsliste, saa hvert [Team] kommer tilbake med `members = null`. Det er
     * riktig: medlemmene er ikke hentet, og ingen skal tro noe annet.
     */
    fun list(ctx: Context, onDone: (Utfall<List<Team>>) -> Unit) {
        if (!Auth.isLoggedIn(ctx)) { Api.ui { onDone(Utfall.IkkeInnlogget) }; return }
        Api.io {
            val r = Api.send(ctx, "GET", PATH)
            val u: Utfall<List<Team>> = if (!r.ok) feil(r) else try {
                val arr = JSONArray(r.body)
                Utfall.Ok((0 until arr.length()).mapNotNull { i ->
                    arr.optJSONObject(i)?.let(Team::fraServer)?.takeIf { it.id.isNotEmpty() }
                })
            } catch (e: Exception) {
                // Et svar vi ikke forstaar er en feil, ikke et tomt lag.
                Log.w(TAG, "kunne ikke lese lagliste", e)
                Utfall.Feil(r.code, retryable = false)
            }
            Api.ui { onDone(u) }
        }
    }

    /**
     * `GET /v1/teams/{id}` — laget med medlemsliste og roller.
     *
     * 404 betyr her «du er ikke medlem» *eller* «laget finnes ikke»: serveren
     * skiller dem ikke, med vilje (`teams.py:_medlemskap` — vi avsloerer ikke at
     * et lag finnes for noen som staar utenfor). Kalleren maa derfor ikke
     * formulere 404 som «laget er slettet».
     */
    fun details(ctx: Context, teamId: String, onDone: (Utfall<Team>) -> Unit) {
        if (!Auth.isLoggedIn(ctx)) { Api.ui { onDone(Utfall.IkkeInnlogget) }; return }
        Api.io {
            val r = Api.send(ctx, "GET", "$PATH/$teamId")
            val u: Utfall<Team> = if (!r.ok) feil(r) else try {
                val t = Team.fraServer(JSONObject(r.body))
                if (t.id.isEmpty()) Utfall.Feil(r.code, false) else Utfall.Ok(t)
            } catch (e: Exception) {
                Log.w(TAG, "kunne ikke lese lag $teamId", e)
                Utfall.Feil(r.code, retryable = false)
            }
            Api.ui { onDone(u) }
        }
    }

    // ---------- Oppretting ----------

    /**
     * `POST /v1/teams`. Oppretteren blir alltid medlem; [iAmLeader] avgjoer om
     * hen blir lagleder eller om laget staar uten leder til noen tar rollen
     * (§4 «jeg er leder» / «opprett for leder»).
     *
     * Navnet er grenset til 64 tegn av serveren, og grensen haandheves her:
     * et for langt navn gir 422, og 422 er ikke `retryable`, saa det ville sett
     * ut som en helt annen feil enn den er (`android/CLAUDE.md`).
     */
    const val NAME_MAX = 64

    // ---------- Invitasjon (§4) ----------

    /**
     * Svaret fra `POST /{id}/invite`.
     *
     * **Et 201 betyr ikke at invitasjonen kom fram.** Serveren lagrer raden og
     * *forsoeker* aa sende; lyktes det ikke, staar [levert] som `false` og
     * [feil] sier hvorfor. SMS sendes ikke i det hele tatt (utsatt til v2), saa
     * telefonnummer gir alltid `levert = false`.
     *
     * I begge tilfeller faar klienten [url], og det er med vilje: lenken kan
     * deles manuelt. En invitasjon som ikke ble sendt er derfor ikke tapt — men
     * den maa vises som noe brukeren maa gjoere noe med, ikke som en kvittering.
     */
    class Invitasjon(val url: String, val maal: String,
                     val levert: Boolean, val feil: String)

    fun invite(ctx: Context, teamId: String, epostEllerTlf: String,
               onDone: (Utfall<Invitasjon>) -> Unit) {
        if (!Auth.isLoggedIn(ctx)) { Api.ui { onDone(Utfall.IkkeInnlogget) }; return }
        Api.io {
            val r = Api.postJson(ctx, "$PATH/$teamId/invite", JSONObject().apply {
                put("email_or_phone", epostEllerTlf.trim())
            })
            val u: Utfall<Invitasjon> = if (!r.ok) feil(r) else try {
                val o = JSONObject(r.body)
                Utfall.Ok(Invitasjon(
                    url = o.optString("url", ""),
                    maal = o.optString("target", epostEllerTlf.trim()),
                    levert = o.optString("delivery_status") == "sent",
                    feil = if (o.isNull("delivery_error")) ""
                           else o.optString("delivery_error", ""),
                ))
            } catch (e: Exception) {
                Log.w(TAG, "kunne ikke lese invitasjonssvar", e)
                Utfall.Feil(r.code, retryable = false)
            }
            Api.ui { onDone(u) }
        }
    }

    // ---------- Fjern medlem (§11) ----------

    /**
     * `DELETE /{team_id}/members/{member_id}` — krever lagleder, svarer 204.
     *
     * [memberId] er den **interne** `user_id` fra `members[]`, ikke `public_id`
     * (`backend/KONTRAKT.md` §6).
     */
    fun removeMember(ctx: Context, teamId: String, memberId: String,
                     onDone: (Utfall<Unit>) -> Unit) {
        if (!Auth.isLoggedIn(ctx)) { Api.ui { onDone(Utfall.IkkeInnlogget) }; return }
        Api.io {
            val r = Api.send(ctx, "DELETE", "$PATH/$teamId/members/$memberId")
            val u: Utfall<Unit> = if (r.ok) Utfall.Ok(Unit) else feil(r)
            Api.ui { onDone(u) }
        }
    }

    fun create(ctx: Context, name: String, kind: String, iAmLeader: Boolean,
               onDone: (Utfall<Team>) -> Unit) {
        if (!Auth.isLoggedIn(ctx)) { Api.ui { onDone(Utfall.IkkeInnlogget) }; return }
        val n = name.trim().take(NAME_MAX)
        Api.io {
            val r = Api.postJson(ctx, PATH, JSONObject().apply {
                put("name", n)
                put("kind", kind)
                put("i_am_leader", iAmLeader)
            })
            val u: Utfall<Team> = if (!r.ok) feil(r) else try {
                val t = Team.fraServer(JSONObject(r.body))
                if (t.id.isEmpty()) Utfall.Feil(r.code, false) else Utfall.Ok(t)
            } catch (e: Exception) {
                // Laget kan vaere opprettet selv om svaret ikke lot seg lese.
                // Derfor «kunne ikke bekreftes», ikke «ble ikke opprettet» —
                // og listen hentes paa nytt av kalleren.
                Log.w(TAG, "lag opprettet, men svaret kunne ikke leses", e)
                Utfall.Feil(r.code, retryable = false)
            }
            Api.ui { onDone(u) }
        }
    }
}
