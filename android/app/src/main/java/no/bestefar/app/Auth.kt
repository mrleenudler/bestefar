package no.bestefar.app

import android.content.Context
import android.util.Log
import org.json.JSONObject

/**
 * Innloggingsøkten på klientsiden (backend_spec §1).
 *
 * Backenden gir to tokener, og de er *ikke* utskiftbare:
 *
 *  - **access-token** — en statsløs JWT med 60 minutters levetid. Den kan ikke
 *    tilbakekalles. Det er et bevisst valg, ikke en mangel, men det legger et
 *    krav på oss: `/v1/auth/logout` gjør den ikke ugyldig. Verifisert mot
 *    produksjon — etter 204 fra `/logout` svarer `GET /v1/profile` fortsatt 200
 *    med det samme tokenet. **Klienten må slette tokenene selv**, ellers har en
 *    utlogget bruker full tilgang i opptil en time.
 *  - **refresh-token** — ugjennomsiktig, lagret hashet hos oss, og det
 *    **roteres ved hver bruk**. Dukker et allerede brukt token opp igjen,
 *    tolker serveren det som en lekkasje og tilbakekaller ALLE brukerens økter.
 *    To parallelle `/refresh` logger altså brukeren ut. Derfor er [refresh]
 *    `@Synchronized` og eneste vei inn.
 *
 * Brukeren skal forbli innlogget på ubestemt tid: så lenge appen brukes
 * innimellom, fornyes refresh-tokenet før det utløper. Feiler fornyelsen, er
 * konsekvensen ordinær innlogging — **aldri** tap av lokale data. Alt appen
 * gjør uten konto virker uendret.
 */
object Auth {

    private const val TAG = "BestefarAuth"

    private const val K_ACCESS = "access_token"
    private const val K_REFRESH = "refresh_token"

    /**
     * Marg før utløp. Et token som utløper om ett minutt er i praksis utløpt:
     * en opplasting kan bruke lengre tid enn det, og et 401 midt i en
     * multipart-strøm koster mer enn en fornyelse i forkant.
     */
    private const val EXPIRY_MARGIN_MS = 5 * 60_000L

    // ---------- Lagring ----------

    fun accessToken(ctx: Context): String {
        migrateLegacy(ctx)
        return Secrets.get(ctx, K_ACCESS)
    }

    fun refreshToken(ctx: Context): String = Secrets.get(ctx, K_REFRESH)

    fun isLoggedIn(ctx: Context): Boolean = refreshToken(ctx).isNotEmpty()

    /**
     * Runde 12 la access-tokenet i klartekst i `bestefar_ui`. Den fila går inn i
     * sikkerhetskopien (Store.exportPrefs er generisk over alle nøkler), så
     * verdien flyttes hit og slettes der. Kjøres én gang og er så en no-op.
     */
    @Suppress("DEPRECATION")
    private fun migrateLegacy(ctx: Context) {
        val store = Store.get(ctx)
        val old = store.legacyAuthToken
        if (old.isEmpty()) return
        Secrets.put(ctx, K_ACCESS, old)
        store.legacyAuthToken = ""
    }

    /** Lagrer svaret fra auth-endepunktene (google, apple, email/verify, refresh). */
    fun saveSession(ctx: Context, body: String): Boolean {
        val o = try { JSONObject(body) } catch (_: Exception) { return false }
        val access = o.optString("access_token", "")
        if (access.isEmpty()) return false
        Secrets.put(ctx, K_ACCESS, access)
        o.optString("refresh_token", "").takeIf { it.isNotEmpty() }
            ?.let { Secrets.put(ctx, K_REFRESH, it) }
        val store = Store.get(ctx)
        store.authExpiresAt = System.currentTimeMillis() + o.optLong("expires_in", 3600) * 1000L
        store.accountPublicId = o.optString("public_id", store.accountPublicId)
        store.accountName = o.optString("display_name", store.accountName)
        return true
    }

    /** Sletter økten lokalt. Kalles alltid til slutt i [logout], uansett svar. */
    fun clearLocal(ctx: Context) {
        Secrets.remove(ctx, K_ACCESS, K_REFRESH)
        val store = Store.get(ctx)
        store.authExpiresAt = 0L
        store.accountPublicId = ""
        store.accountName = ""
        // Merk: backupCode og lokale data røres IKKE. Å logge ut er ikke å
        // slette seg selv, og en app som mister loggen ved utlogging er en app
        // ingen tør logge ut av.
    }

    // ---------- Fornyelse ----------

    /** Er access-tokenet i ferd med å gå ut (eller borte)? */
    fun needsRefresh(ctx: Context): Boolean {
        if (!isLoggedIn(ctx)) return false
        if (accessToken(ctx).isEmpty()) return true
        return System.currentTimeMillis() > Store.get(ctx).authExpiresAt - EXPIRY_MARGIN_MS
    }

    /**
     * Fornyer økten. **`@Synchronized` er sikkerhetskritisk, ikke pynt:** to
     * samtidige kall ville sendt det samme refresh-tokenet to ganger, og
     * serveren tolker gjenbruk som en lekkasje og logger ut alle enhetene.
     *
     * Låsen dekker også tilfellet der to tråder ventet på den samme
     * fornyelsen — den andre ser da at tokenet allerede er nytt og gjør
     * ingenting.
     */
    @Synchronized
    fun refresh(ctx: Context): Boolean {
        val token = refreshToken(ctx)
        if (token.isEmpty()) return false
        // Rakk noen andre å fornye mens vi ventet på låsen?
        if (!needsRefresh(ctx)) return true

        val resp = Api.postJson(ctx, "/v1/auth/refresh",
            JSONObject().put("refresh_token", token), authRetry = false)
        return when {
            resp.ok -> saveSession(ctx, resp.body)
            // 401 = tokenet er dødt eller økten tilbakekalt. Ny innlogging er
            // eneste vei videre; å beholde et token vi vet er ugyldig ville
            // bare gitt et nytt 401 ved hver forespørsel.
            resp.code == 401 -> { clearLocal(ctx); false }
            // Nett nede / 5xx: behold tokenet og prøv igjen senere.
            else -> { Log.d(TAG, "Fornyelse utsatt (${resp.code})"); false }
        }
    }

    // ---------- Utlogging ----------

    /**
     * Ekte utlogging. Rekkefølgen er valgt slik at det som *bare* kan gjøres
     * mens vi fortsatt har et gyldig token, gjøres først:
     *
     *  1. `/v1/devices/unregister` — ellers fortsetter telefonen å få varsler
     *     som var ment for den utloggede brukeren.
     *  2. `/v1/auth/logout` — tilbakekaller refresh-tokenet på serveren.
     *  3. Lokal sletting — **skjer uansett**, også når nettet er nede. Alt
     *     annet ville latt «logg ut» feile stille og etterlatt et gyldig
     *     access-token på telefonen.
     *
     * Blokkerende; kall via [Api.io].
     */
    fun logout(ctx: Context) {
        val push = Store.get(ctx).pushToken
        if (push.isNotEmpty()) {
            Api.postJson(ctx, "/v1/devices/unregister",
                JSONObject().put("push_token", push), authRetry = false)
        }
        val token = refreshToken(ctx)
        if (token.isNotEmpty()) {
            Api.postJson(ctx, "/v1/auth/logout",
                JSONObject().put("refresh_token", token), authRetry = false)
        }
        clearLocal(ctx)
    }
}
