package no.bestefar.app

import android.app.Activity
import android.content.Context
import android.util.Log
import androidx.credentials.CredentialManager
import androidx.credentials.CredentialManagerCallback
import androidx.credentials.CustomCredential
import androidx.credentials.GetCredentialRequest
import androidx.credentials.GetCredentialResponse
import androidx.credentials.exceptions.GetCredentialCancellationException
import androidx.credentials.exceptions.GetCredentialException
import androidx.credentials.exceptions.NoCredentialException
import com.google.android.libraries.identity.googleid.GetSignInWithGoogleOption
import com.google.android.libraries.identity.googleid.GoogleIdTokenCredential
import org.json.JSONObject
import java.util.concurrent.Executors

/**
 * Innlogging (backend_spec §1).
 *
 * Modellen er den vanlige: leverandøren gir oss et ID-token, vi bytter det inn
 * i **våre egne** tokener hos backenden, og alt videre går på dem. [Auth] eier
 * øktene; denne fila eier bare veien inn.
 *
 * **Credential Manager, ikke Google Sign-In.** `GoogleSignInClient` er utfaset,
 * og en ny integrasjon mot et utfaset API er arbeid som må gjøres om igjen.
 * Credential Manager er dessuten ett API for flere kontotyper, så passnøkler
 * eller lagrede passord kan legges til senere uten en ny innloggingsflyt.
 *
 * **Ingen konto er fortsatt et gyldig valg.** Appen er offline-først: scan,
 * innsikt, serier og jaktlogg virker uendret uten innlogging. Kontoen kjøper
 * sikkerhetskopi til server, venner og lag — ingenting annet. Det står i
 * skjermen, og ingenting her spør brukeren uoppfordret.
 */
object Login {

    private const val TAG = "BestefarLogin"

    /** Egen tråd: Credential Manager kaller tilbake på den vi gir den. */
    private val executor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "bestefar-login").apply { isDaemon = true }
    }

    /**
     * Utfallet av et innloggingsforsøk. [Avbrutt] er med vilje skilt fra
     * [Feil]: en bruker som trykker «tilbake» i kontovelgeren skal ikke få en
     * feilmelding om noe de gjorde med vilje.
     */
    sealed class Result {
        /**
         * [nyKonto] er serverens `is_new`: kontoen ble opprettet av nettopp
         * dette kallet. Det er det eneste oeyeblikket vi med sikkerhet vet at
         * brukeren ikke har en sikkerhetskopi noe sted, og derfor det ene
         * riktige tidspunktet aa tilby en.
         */
        class Ok(val nyKonto: Boolean) : Result()
        object Avbrutt : Result()
        class Feil(val melding: String) : Result()
    }

    // ---------- Google ----------

    /**
     * Web-klient-ID-en (`client_type: 3` i google-services.json). Det er DEN
     * som blir `aud` i ID-tokenet, og dermed den som må stå i backendens
     * `GOOGLE_CLIENT_IDS` — ikke Android-klient-ID-en. Forveksling her gir et
     * gyldig token som backenden avviser, og feilmeldingen sier ingenting om
     * hvorfor.
     */
    private fun webClientId(): String = BuildConfig.GOOGLE_WEB_CLIENT_ID

    /** Google-knappen skjules helt når appen er bygget uten klient-ID. */
    fun googleConfigured(): Boolean = webClientId().isNotEmpty()

    /**
     * «Fortsett med Google». Bruker `GetSignInWithGoogleOption` — altså den
     * eksplisitte knappeflyten, ikke den filtrerte bunnarken. Forskjellen har
     * betydning: bunnarken feiler for en bruker som aldri har logget inn i
     * appen før, og det er nøyaktig den brukeren en «logg inn»-knapp finnes for.
     *
     * [onDone] kalles alltid, på UI-tråden.
     */
    fun withGoogle(a: Activity, onDone: (Result) -> Unit) {
        if (!googleConfigured()) {
            onDone(Result.Feil(a.getString(R.string.login_google_missing))); return
        }
        val request = GetCredentialRequest.Builder()
            .addCredentialOption(
                GetSignInWithGoogleOption.Builder(webClientId()).build())
            .build()

        CredentialManager.create(a).getCredentialAsync(
            a, request, null, executor,
            object : CredentialManagerCallback<GetCredentialResponse, GetCredentialException> {
                override fun onResult(result: GetCredentialResponse) {
                    val token = idToken(result)
                    if (token == null) {
                        Api.ui { onDone(Result.Feil(a.getString(R.string.login_no_token))) }
                        return
                    }
                    // Vi er allerede utenfor UI-tråden, men bytt til Apis egen
                    // kø slik at innloggingskallet står i samme rekkefølge som
                    // resten av nettverkstrafikken.
                    Api.io { exchange(a, "/v1/auth/google", token, onDone) }
                }

                override fun onError(e: GetCredentialException) {
                    Log.d(TAG, "Google-innlogging feilet: ${e::class.simpleName} ${e.message}")
                    Api.ui {
                        onDone(when (e) {
                            is GetCredentialCancellationException -> Result.Avbrutt
                            // Ingen Google-konto på telefonen. Det er ikke en
                            // feil i appen, og brukeren skal få vite hva de kan
                            // gjøre i stedet.
                            is NoCredentialException ->
                                Result.Feil(a.getString(R.string.login_no_account))
                            else -> Result.Feil(a.getString(R.string.login_google_failed))
                        })
                    }
                }
            })
    }

    private fun idToken(result: GetCredentialResponse): String? {
        val cred = result.credential
        if (cred is CustomCredential &&
            cred.type == GoogleIdTokenCredential.TYPE_GOOGLE_ID_TOKEN_CREDENTIAL) {
            return try {
                GoogleIdTokenCredential.createFrom(cred.data).idToken
            } catch (e: Exception) {
                Log.e(TAG, "Kunne ikke lese ID-tokenet", e); null
            }
        }
        return null
    }

    // ---------- E-postkode ----------

    /**
     * Ber om en engangskode. Svaret er **alltid** 202, også for en adresse vi
     * ikke kjenner — et svar som skilte kjent fra ukjent ville gjort
     * endepunktet til et oppslagsverk over hvem som bruker appen.
     *
     * Returnerer sekunder til «send ny kode» kan brukes (serveren bestemmer,
     * klienten teller bare ned), eller -1 ved feil.
     */
    fun startEmail(ctx: Context, epost: String, onDone: (Result, Int) -> Unit) {
        Api.io {
            val resp = Api.postJson(ctx, "/v1/auth/email/start",
                JSONObject().put("email", epost), authRetry = false)
            val cooldown = try {
                JSONObject(resp.body).optInt("resend_after_seconds", 60)
            } catch (_: Exception) { 60 }
            Api.ui {
                when {
                    // Ok(false): aa be om en kode oppretter ingen konto. Om det
                    // blir en ny konto avgjoeres foerst i /email/verify.
                    resp.ok -> onDone(Result.Ok(nyKonto = false), cooldown)
                    // 429 er sperrefristen på serveren, ikke en feil brukeren
                    // har gjort. Fristen står i svaret.
                    resp.code == 429 -> onDone(
                        Result.Feil(ctx.getString(R.string.login_email_too_soon)), cooldown)
                    resp.code == 0 -> onDone(
                        Result.Feil(ctx.getString(R.string.login_offline)), -1)
                    resp.code == 422 -> onDone(
                        Result.Feil(ctx.getString(R.string.login_email_invalid)), -1)
                    else -> onDone(Result.Feil(
                        ctx.getString(R.string.login_failed, resp.code)), -1)
                }
            }
        }
    }

    fun verifyEmail(ctx: Context, epost: String, kode: String, onDone: (Result) -> Unit) {
        Api.io {
            val resp = Api.postJson(ctx, "/v1/auth/email/verify",
                JSONObject().put("email", epost).put("code", kode), authRetry = false)
            finish(ctx, resp, onDone)
        }
    }

    // ---------- Felles ----------

    /** Blokkerende; kalles fra [Api.io]. */
    private fun exchange(ctx: Context, path: String, idToken: String,
                         onDone: (Result) -> Unit) {
        val resp = Api.postJson(ctx, path,
            JSONObject().put("id_token", idToken), authRetry = false)
        finish(ctx, resp, onDone)
    }

    private fun finish(ctx: Context, resp: Api.Resp, onDone: (Result) -> Unit) {
        val ok = resp.ok && Auth.saveSession(ctx, resp.body)
        // `is_new` staar i svaret fra alle tre innloggingsveiene
        // (TokenParNyBruker i contracts/openapi.json). Mangler feltet, antar vi
        // eksisterende konto: aa tilby en foerstegangs-sikkerhetskopi til noen
        // som allerede har en, er verre enn aa la vaere.
        val ny = ok && try {
            JSONObject(resp.body).optBoolean("is_new", false)
        } catch (_: Exception) { false }
        Api.ui {
            onDone(when {
                ok -> Result.Ok(ny)
                // 401 fra oss betyr at leverandørens token ble avvist — nesten
                // alltid feil `aud`, altså at Android-klient-ID-en er brukt der
                // web-klient-ID-en skulle stått, eller at backenden mangler
                // GOOGLE_CLIENT_IDS.
                resp.code == 401 -> Result.Feil(ctx.getString(R.string.login_rejected))
                resp.code == 400 || resp.code == 422 ->
                    Result.Feil(ctx.getString(R.string.login_bad_code))
                resp.code == 429 -> Result.Feil(ctx.getString(R.string.login_email_too_soon))
                resp.code == 503 -> Result.Feil(ctx.getString(R.string.login_not_configured))
                resp.code == 0 -> Result.Feil(ctx.getString(R.string.login_offline))
                else -> Result.Feil(ctx.getString(R.string.login_failed, resp.code))
            })
        }
    }

    /**
     * Lagrer visningsnavnet (`PUT /v1/profile`).
     *
     * **Moderasjonen kjører synkront på serveren**, så svaret er endelig i det
     * øyeblikket det kommer: 200 betyr godkjent og lagret, 422 betyr avvist og
     * *ikke* lagret. Det finnes ingen mellomtilstand å vise —
     * `moderation.review` returnerer bare `approved` eller `rejected`, aldri
     * `pending`, og den manuelle køen finnes ikke (ÅP-B8). Klienten skal derfor
     * aldri si «venter på moderasjon».
     *
     * [onDone] får serverens normaliserte navn ved suksess, eller
     * **begrunnelsen** ved avvisning — den er skrevet for å vises til brukeren
     * (`moderation.review`, «Begrunnelsen er ment aa vises for brukeren»), så vi
     * viser den ordrett framfor å finne på vår egen.
     */
    fun lagreVisningsnavn(ctx: Context, navn: String,
                          onDone: (godkjent: Boolean, tekst: String) -> Unit) {
        Api.io {
            val kropp = JSONObject().put("display_name", navn)
                .toString().toByteArray(Charsets.UTF_8)
            val resp = Api.send(ctx, "PUT", "/v1/profile",
                "application/json; charset=utf-8", kropp)
            val tekst = when {
                // Serveren normaliserer navnet (trimmer, kollapser mellomrom),
                // saa vi tar imot DET og ikke det brukeren skrev.
                resp.ok -> try {
                    JSONObject(resp.body).optString("display_name", navn)
                } catch (_: Exception) { navn }
                // FastAPI pakker HTTPException(422, begrunnelse) som
                // {"detail": "<begrunnelse>"}.
                resp.code == 422 -> try {
                    JSONObject(resp.body).optString("detail", "")
                } catch (_: Exception) { "" }
                else -> ""
            }
            Api.ui { onDone(resp.ok, tekst) }
        }
    }

    /**
     * Utlogging. [Auth.logout] gjør selve arbeidet; her legger vi bare på det
     * som er UI-ets ansvar: låsen glemmer at den er åpnet, slik at neste
     * bruker av telefonen ikke arver en gyldig frist.
     */
    fun logout(ctx: Context, onDone: () -> Unit) {
        Api.io {
            Auth.logout(ctx)
            Lock.forget()
            Api.ui { onDone() }
        }
    }
}
