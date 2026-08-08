package no.bestefar.app

import android.content.Context
import android.util.Log
import com.google.android.gms.auth.blockstore.Blockstore
import com.google.android.gms.auth.blockstore.RetrieveBytesRequest
import com.google.android.gms.auth.blockstore.StoreBytesData
import com.google.android.gms.tasks.Tasks
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/**
 * Nøkkelforvaltning for sikkerhetskopien (musingsUI runde 13).
 *
 * Runde 12 løste kryptering og serialisering, men lot brukeren sitte igjen med
 * hele ansvaret: en kode på tjue tegn som måtte skrives ned, ellers var kopien
 * tapt. Det er riktig kryptografi og feil produkt — folk skriver ikke ned
 * koden, og oppdager det den dagen telefonen er borte.
 *
 * Denne runden gir tre veier inn til den samme nøkkelen, i rekkefølge:
 *
 *  1. **Block Store** (denne fila). Google Play lagrer noen få byte for oss og
 *     tar dem med til den nye telefonen ved gjenoppretting. Med
 *     `isEndToEndEncryptionAvailable()` er de kryptert med en nøkkel avledet av
 *     skjermlåsen — Google kan ikke lese dem, og vi trenger ikke stole på at de
 *     lar være. Da er gjenoppretting ett trykk, uten kode.
 *  2. **Gjenopprettingskoden**, degradert til nødutgang. Den finnes fortsatt,
 *     men vises ikke lenger som en dialog man må forholde seg til ved første
 *     kopi. Den ligger under Avanserte innstillinger → Sikkerhetskopi.
 *  3. **Deponering hos oss** — bare hvis brukeren uttrykkelig ber om det, se
 *     [escrowPut]. Da kan vi låse opp kopien, og det står i klartekst i
 *     hjelpeteksten. Av som standard.
 *
 * Alt her er blokkerende og skal kalles via [Api.io].
 */
object BackupKeys {

    private const val TAG = "BestefarKeys"
    private const val BLOCK_KEY = "no.bestefar.backup.code"
    private const val WAIT_S = 8L

    // ---------- Block Store ----------

    /**
     * Er Block Store-innholdet ende-til-ende-kryptert på denne enheten?
     * Krever skjermlås. Er det ikke det, lagrer vi ingenting: en nøkkel som
     * ligger lesbar hos en tredjepart er dårligere enn ingen nøkkel der.
     */
    fun e2eAvailable(ctx: Context): Boolean = try {
        Tasks.await(Blockstore.getClient(ctx).isEndToEndEncryptionAvailable(),
            WAIT_S, TimeUnit.SECONDS)
    } catch (e: Exception) {
        Log.d(TAG, "Block Store utilgjengelig: ${e.message}")
        false
    }

    /** Lagrer koden i Block Store. Returnerer om det gikk. */
    fun store(ctx: Context, code: String): Boolean {
        if (code.isEmpty()) return false
        if (!e2eAvailable(ctx)) return false
        return try {
            Tasks.await(Blockstore.getClient(ctx).storeBytes(
                StoreBytesData.Builder()
                    .setBytes(code.toByteArray(Charsets.UTF_8))
                    .setKey(BLOCK_KEY)
                    // Uten dette overlever nøkkelen kun en enhet-til-enhet-
                    // overføring. Poenget her er telefonen som ALDRI kommer
                    // tilbake, så skyen er hele vitsen.
                    .setShouldBackupToCloud(true)
                    .build()),
                WAIT_S, TimeUnit.SECONDS)
            true
        } catch (e: Exception) {
            Log.d(TAG, "Kunne ikke lagre i Block Store: ${e.message}")
            false
        }
    }

    /** Henter koden fra Block Store, eller tom streng. */
    fun retrieve(ctx: Context): String = try {
        val resp = Tasks.await(Blockstore.getClient(ctx).retrieveBytes(
            RetrieveBytesRequest.Builder().setKeys(listOf(BLOCK_KEY)).build()),
            WAIT_S, TimeUnit.SECONDS)
        resp.blockstoreDataMap[BLOCK_KEY]?.bytes?.toString(Charsets.UTF_8) ?: ""
    } catch (e: Exception) {
        Log.d(TAG, "Ingen nøkkel i Block Store: ${e.message}")
        ""
    }

    // ---------- Deponering hos serveren (valgfritt) ----------

    private const val ESCROW = "/v1/backup/key-escrow"

    private fun b64(s: String) =
        android.util.Base64.encodeToString(s.toByteArray(Charsets.UTF_8),
            android.util.Base64.NO_WRAP)

    /**
     * Legger nøkkelen hos backenden (backend_spec §2.1). Kalles KUN når
     * brukeren har slått på «Gjenopprett uten kode» og dermed sagt ja til at vi
     * kan låse opp kopien.
     *
     * `key_material` er base64 av ugjennomsiktige byte — serveren bryr seg ikke
     * om det er en nøkkel eller en kode. Vi sender koden som ASCII, siden det
     * er den nøkkelen faktisk utledes fra.
     *
     * 503 betyr at deponering ikke er konfigurert på serveren. Da lagres
     * ingenting, og det er riktig: alternativet ville vært en nøkkel i
     * klartekst som nødløsning.
     */
    fun escrowPut(ctx: Context, code: String): Api.Resp =
        Api.send(ctx, "PUT", ESCROW, "application/json; charset=utf-8",
            JSONObject().put("key_material", b64(code))
                .toString().toByteArray(Charsets.UTF_8))

    fun escrowGet(ctx: Context): Pair<Api.Resp, String> {
        val resp = Api.send(ctx, "GET", ESCROW)
        val code = if (resp.ok) try {
            val raw = JSONObject(resp.body).optString("key_material", "")
            if (raw.isEmpty()) "" else String(
                android.util.Base64.decode(raw, android.util.Base64.DEFAULT), Charsets.UTF_8)
        } catch (_: Exception) { "" } else ""
        return resp to code
    }

    fun escrowDelete(ctx: Context): Api.Resp = Api.send(ctx, "DELETE", ESCROW)

    // ---------- Samlet oppslag ----------

    /**
     * Nøkkelen appen skal bruke nå, i prioritert rekkefølge: den vi allerede
     * har lokalt, så Block Store, så deponeringen. Finner vi den et annet sted
     * enn lokalt, skrives den lokalt — neste gang er den ett oppslag unna.
     *
     * Returnerer tom streng når ingen kilde har noe. Da (og bare da) må
     * brukeren taste koden.
     */
    fun resolve(ctx: Context, deponertPaaServer: Boolean = false): String {
        val st = Store.get(ctx)
        st.backupCode.takeIf { it.isNotEmpty() }?.let { return it }

        retrieve(ctx).takeIf { it.isNotEmpty() }?.let {
            st.backupCode = it
            return it
        }
        // [deponertPaaServer] kommer fra `escrowed` i GET /v1/backup/meta, og er
        // ikke det samme som den lokale bryteren.
        //
        // DETTE ER POENGET: `st.backupEscrow` ligger i prefs, og prefs er borte
        // etter en reinstallasjon. Paa en ny telefon staar bryteren derfor av
        // selv om noekkelen ligger deponert hos oss - og da hoppet vi over
        // deponeringen i noeyaktig det scenarioet den finnes for. Serverens
        // `escrowed` overlever reinstallasjonen; den lokale bryteren gjoer det
        // ikke.
        if ((st.backupEscrow || deponertPaaServer) && Auth.isLoggedIn(ctx)) {
            val (resp, code) = escrowGet(ctx)
            if (resp.ok && code.isNotEmpty()) {
                st.backupCode = code
                store(ctx, code)          // og legg den i Block Store med det samme
                // Bryteren gjenspeiler naa virkeligheten igjen: brukeren HAR
                // deponert, og innstillingssiden skal ikke paastaa noe annet.
                st.backupEscrow = true
                return code
            }
            // MERK: her forsvinner forskjellen paa «deponeringen er tom» (404)
            // og «vi naadde ikke fram» (0, 405, 5xx). Begge ender med at
            // brukeren blir bedt om koden, som er trygt, men som saa ut som en
            // manglende deponering da GET i praksis var POST og alltid ga 405.
            // Derfor logges alt som ikke er 404 - det er den eneste sporen.
            if (resp.code != 404) Log.w(TAG, "Deponeringen svarte ${resp.code}")
        }
        return ""
    }

    /**
     * Sørger for at det FINNES en nøkkel, og at den er lagret alle stedene
     * brukeren har sagt ja til. Kalles før hver sikkerhetskopi.
     */
    fun ensure(ctx: Context): String {
        val st = Store.get(ctx)
        val code = resolve(ctx).ifEmpty {
            Backup.newRecoveryCode().also { st.backupCode = it }
        }
        store(ctx, code)
        if (st.backupEscrow && Auth.isLoggedIn(ctx)) escrowPut(ctx, code)
        return code
    }
}
