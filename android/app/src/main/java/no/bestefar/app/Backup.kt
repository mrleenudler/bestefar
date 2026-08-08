package no.bestefar.app

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.security.SecureRandom
import java.time.Instant
import javax.crypto.Cipher
import javax.crypto.SecretKeyFactory
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.PBEKeySpec
import javax.crypto.spec.SecretKeySpec

/**
 * Klient-kryptert sikkerhetskopi (backend_spec.md §2).
 *
 * Serveren lagrer BYTES DEN IKKE KAN LESE. Hele kryptering-, serialiserings- og
 * gjenopprettingsjobben er derfor klientside, og den kan bygges og testes uten
 * at innloggingen er på plass — opplastingen er bare siste steg.
 *
 * ## Nøkkelen
 *
 * Det vanskelige her er ikke kryptoen, det er NØKKELHÅNDTERINGEN. En nøkkel som
 * bare finnes på telefonen er verdiløs i det scenarioet kopien eksisterer for:
 * telefonen er borte. Nøkkelen må derfor kunne gjenskapes fra noe brukeren har
 * utenfor telefonen.
 *
 * Vi genererer en GJENOPPRETTINGSKODE på 20 tegn (100 bit) som vises én gang og
 * må skrives ned. Nøkkelen utledes med PBKDF2-HMAC-SHA256 fra koden. Passord
 * valgt av brukeren ble bevisst vraket: en kopi som er kryptert med «Bestefar1»
 * gir en falsk trygghet, og her finnes ingen server som kan bremse gjetting —
 * angriperen har hele bloben.
 *
 * Prisen er ærlig og må stå i UI-et: **mister du koden, er kopien tapt.** Det
 * er en direkte følge av at serveren ikke kan lese bloben, ikke noe vi kan
 * konstruere oss ut av.
 *
 * ## Blobformat (versjon 1)
 *
 *     "BFBK" | 1 byte versjon | 16 byte salt | 12 byte IV | GCM-chiffertekst
 *
 * Salt og IV ligger i klartekst; det skal de. Begge er offentlige per
 * konstruksjon, og et nytt salt per kopi hindrer at én utledet nøkkel kan
 * gjenbrukes mot flere kopier.
 */
object Backup {

    private const val TAG = "BestefarBackup"

    private val MAGIC = byteArrayOf('B'.code.toByte(), 'F'.code.toByte(),
        'B'.code.toByte(), 'K'.code.toByte())
    private const val VERSION = 1
    private const val SALT_LEN = 16
    private const val IV_LEN = 12
    private const val TAG_BITS = 128
    /**
     * PBKDF2-runder. 210 000 er OWASPs anbefaling for HMAC-SHA256 (2023) og
     * koster ~0,2–0,5 s på en middels telefon — merkbart nok til at brukeren
     * skjønner at noe skjer, billig nok til at det ikke irriterer.
     */
    private const val ITERATIONS = 210_000
    private const val KEY_BITS = 256

    /** Formatversjon på selve INNHOLDET (uavhengig av blobformatet). */
    private const val SNAPSHOT_VERSION = 1

    /**
     * Alfabetet i gjenopprettingskoden: Crockford base32 uten I, L, O og U.
     * I/L/O forveksles med 1/0 når koden leses opp eller skrives av for hånd;
     * U er ute for å unngå at tilfeldige bokstavrekker danner stygge ord.
     */
    private const val ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    private const val CODE_LEN = 20

    // ---------- Gjenopprettingskode ----------

    /** Ny kode, vist som «XXXXX-XXXXX-XXXXX-XXXXX». 20 tegn a 5 bit = 100 bit. */
    fun newRecoveryCode(): String {
        val rnd = SecureRandom()
        val sb = StringBuilder(CODE_LEN)
        repeat(CODE_LEN) { sb.append(ALPHABET[rnd.nextInt(ALPHABET.length)]) }
        return sb.toString()
    }

    fun formatCode(code: String): String =
        code.chunked(5).joinToString("-")

    /**
     * Normaliser en kode brukeren har skrevet inn: bindestreker og mellomrom
     * bort, store bokstaver, og de klassiske lesefeilene foldet inn (I/L→1,
     * O→0). Bedre å godta en åpenbar avskrivningsfeil enn å avvise en riktig
     * kode fordi den ble lest opp over telefon.
     */
    fun normalizeCode(input: String): String =
        input.uppercase()
            .filter { it.isLetterOrDigit() }
            .map { when (it) { 'I', 'L' -> '1'; 'O' -> '0'; else -> it } }
            .joinToString("")

    // ---------- Øyeblikksbilde ----------

    /**
     * Hele brukerens data som én JSON. Seriene og jaktpostene tas med
     * GRAVSTEINER OG ALT ([Store.allSeriesRaw]) — en kopi som utelater
     * slettingene ville gjenoppstå med data brukeren bevisst har slettet.
     */
    fun snapshot(ctx: Context): JSONObject {
        val store = Store.get(ctx)
        return JSONObject().apply {
            put("v", SNAPSHOT_VERSION)
            put("app", BuildConfig.VERSION_NAME)
            put("ts", System.currentTimeMillis())
            put("prefs", store.exportPrefs())
            put("series", JSONArray().apply {
                store.allSeriesRaw().forEach { put(it.toJson()) }
            })
            put("hunts", JSONArray().apply {
                store.allHuntsRaw().forEach { put(it.toJson()) }
            })
        }
    }

    /**
     * Skriv et øyeblikksbilde tilbake. ERSTATTER alt lokalt innhold — dette er
     * «gjenopprett på ny telefon», ikke en fletting. Fletting ville krevd at vi
     * kunne avgjøre hvilken side som er nyest per post, og det er nettopp den
     * regelen §2 legger på klienten (last-write-wins per post-ID); den hører
     * hjemme i synken, ikke i gjenopprettingen.
     */
    fun restore(ctx: Context, snap: JSONObject) {
        val store = Store.get(ctx)
        val series = snap.optJSONArray("series") ?: JSONArray()
        val hunts = snap.optJSONArray("hunts") ?: JSONArray()
        store.replaceAll(
            (0 until series.length()).mapNotNull {
                try { SeriesRecord.fromJson(series.getJSONObject(it)) }
                catch (e: Exception) { Log.w(TAG, "Hopper over ugyldig serie", e); null }
            },
            (0 until hunts.length()).mapNotNull {
                try { HuntRecord.fromJson(hunts.getJSONObject(it)) }
                catch (e: Exception) { Log.w(TAG, "Hopper over ugyldig jaktpost", e); null }
            })
        snap.optJSONObject("prefs")?.let { store.importPrefs(it) }
    }

    // ---------- Kryptering ----------

    private fun deriveKey(code: String, salt: ByteArray): SecretKeySpec {
        val spec = PBEKeySpec(normalizeCode(code).toCharArray(), salt, ITERATIONS, KEY_BITS)
        val factory = SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256")
        return SecretKeySpec(factory.generateSecret(spec).encoded, "AES")
    }

    fun encrypt(plain: ByteArray, code: String): ByteArray {
        val rnd = SecureRandom()
        val salt = ByteArray(SALT_LEN).also { rnd.nextBytes(it) }
        val iv = ByteArray(IV_LEN).also { rnd.nextBytes(it) }
        val cipher = Cipher.getInstance("AES/GCM/NoPadding").apply {
            init(Cipher.ENCRYPT_MODE, deriveKey(code, salt), GCMParameterSpec(TAG_BITS, iv))
        }
        val out = ByteArrayOutputStream()
        out.write(MAGIC)
        out.write(VERSION)
        out.write(salt)
        out.write(iv)
        out.write(cipher.doFinal(plain))
        return out.toByteArray()
    }

    /** Feil kode gir [BadCodeException] — GCM-taggen skiller det fra en korrupt fil. */
    class BadCodeException : Exception("Feil gjenopprettingskode, eller skadet kopi.")

    fun decrypt(blob: ByteArray, code: String): ByteArray {
        val head = MAGIC.size + 1 + SALT_LEN + IV_LEN
        if (blob.size <= head || !blob.copyOfRange(0, MAGIC.size).contentEquals(MAGIC))
            throw IllegalArgumentException("Ikke en Bestefar-sikkerhetskopi.")
        val version = blob[MAGIC.size].toInt()
        if (version != VERSION)
            throw IllegalArgumentException("Ukjent kopiversjon ($version).")
        val salt = blob.copyOfRange(MAGIC.size + 1, MAGIC.size + 1 + SALT_LEN)
        val iv = blob.copyOfRange(MAGIC.size + 1 + SALT_LEN, head)
        val cipher = Cipher.getInstance("AES/GCM/NoPadding").apply {
            init(Cipher.DECRYPT_MODE, deriveKey(code, salt), GCMParameterSpec(TAG_BITS, iv))
        }
        return try {
            cipher.doFinal(blob, head, blob.size - head)
        } catch (e: javax.crypto.AEADBadTagException) {
            throw BadCodeException()
        } catch (e: javax.crypto.BadPaddingException) {
            throw BadCodeException()
        }
    }

    /** Ferdig kryptert blob av dagens data. */
    fun build(ctx: Context, code: String): ByteArray =
        encrypt(snapshot(ctx).toString().toByteArray(Charsets.UTF_8), code)

    fun open(ctx: Context, blob: ByteArray, code: String) =
        restore(ctx, JSONObject(String(decrypt(blob, code), Charsets.UTF_8)))

    // ---------- Opplasting / nedlasting (backend_spec §2) ----------

    /**
     * `PUT /v1/backup` med rå bytes og metadata i query-strengen.
     *
     * `client_ts` er tidsstempelet serveren sammenligner mot den lagrede: en
     * PUT som er ELDRE enn det som ligger der, avvises med 409. Det vernet
     * finnes fordi serveren ikke kan se inn i bloben og derfor ikke kan
     * håndheve last-write-wins per post — uten det kunne en telefon som synker
     * for første gang på måneder viske ut alt som er logget siden.
     *
     * Parameterne er de fire `contracts/openapi.json` deklarerer, og bare de:
     *
     *  - **`client_ts` sendes som ISO-8601 med `Z`**, ikke som epoke-ms.
     *    Serveren tar imot begge (`BackupMeta.client_ts`), men skjemaet
     *    deklarerer `date-time`, og ms-varianten hviler paa at parseren tolker
     *    store heltall som millisekunder og ikke som sekunder. Blir den
     *    tolkningen noen gang strengere, leses tallet som sekunder, tidspunktet
     *    havner titusener av aar fram i tid, og 409-vernet snur — det ville
     *    sluppet gjennom nettopp den utdaterte enheten det finnes for aa stoppe.
     *    **`Z`, aldri `+00:00`:** et `+` i en query-streng leses som mellomrom.
     *  - **`schema_version`** er formatversjonen paa innholdet, altsaa den
     *    serveren gir tilbake i `/meta` slik at en ny telefon kan se om den kan
     *    lese kopien foer den laster ned 16 MB.
     *  - **`app_version` sendes IKKE lenger.** Serveren tar ikke imot en slik
     *    parameter, og ukjente query-parametere forsvinner stille — verdien har
     *    aldri naadd fram. Appversjonen ligger dessuten i bloben (`app`), der
     *    den er til nytte for den som kan dekryptere den.
     *  - **`device_id` sendes fortsatt ikke** — vi har ingen stabil
     *    installasjons-ID aa sette der. AAPNE_PUNKTER ÅP-U13.
     */
    fun upload(ctx: Context, code: String, force: Boolean = false): Api.Resp {
        // Tidsstempelet tas FOER bloben bygges: feltet heter «klientens
        // tidsstempel for oeyeblikksbildet», og noekkelutledningen tar et kvart
        // sekund. Da skal ikke tallet vaere fra etter at bildet ble tatt.
        val naa = Instant.now().toString()
        val blob = build(ctx, code)
        val q = "?client_ts=$naa" +
            "&schema_version=$SNAPSHOT_VERSION" +
            (if (force) "&force=true" else "")
        return Api.send(ctx, "PUT", "/v1/backup$q", "application/octet-stream", blob)
    }

    /**
     * `GET /v1/backup/meta` — «har jeg noe å gjenopprette?» uten å laste ned alt.
     *
     * **Dette kallet hadde ingen kaller før v0.19, og virket ikke.** `Api.send`
     * med `GET` traff serveren som `POST` og fikk 405, og siden ingen kalte det,
     * merket ingen det. Nå gater den gjenopprettingen: 404 herfra betyr at det
     * ikke finnes noe å hente, og da skal vi hverken be om en kode eller laste
     * ned 16 MB for å komme fram til det samme svaret.
     */
    fun meta(ctx: Context): Api.Resp = Api.send(ctx, "GET", "/v1/backup/meta")

    /**
     * Tidspunktet kopien ble laget, som norsk tekst — eller tom streng hvis
     * svaret ikke har et brukbart `client_ts`. Vises i bekreftelsen før en
     * gjenoppretting: «erstatter alt» er en helt annen beslutning når man vet
     * om kopien er fra i går eller fra i fjor.
     */
    fun metaNaar(resp: Api.Resp): String = try {
        val o = JSONObject(resp.body)
        // `client_ts` er nullable i skjemaet, og org.json gir strengen "null"
        // for en JSON-null via optString. Uten isNull-sjekken ville vi sendt
        // "null" videre til parseren og vaert prisgitt at den feiler «riktig».
        if (o.isNull("client_ts")) "" else Ui.norskTid(o.optString("client_ts", ""))
    } catch (_: Exception) {
        ""
    }

    /**
     * `GET /v1/backup` + dekryptering + gjenoppretting. Alt lokalt innhold
     * ERSTATTES ved suksess; feiler dekrypteringen, røres ingenting.
     */
    fun downloadAndRestore(ctx: Context, code: String): Api.Resp {
        val (resp, blob) = Api.download(ctx, "/v1/backup")
        if (!resp.ok || blob == null) return resp
        open(ctx, blob, code)          // kaster BadCodeException ved feil kode
        return resp
    }

    /**
     * Rundtur i minnet: krypter dagens data og dekrypter dem igjen.
     * Verifiserer formatet uten server og uten innlogging — og uten å røre
     * brukerens data, siden vi aldri kaller [restore].
     */
    fun selfTest(ctx: Context): String = try {
        val code = newRecoveryCode()
        val before = snapshot(ctx).toString()
        val blob = encrypt(before.toByteArray(Charsets.UTF_8), code)
        val after = String(decrypt(blob, code), Charsets.UTF_8)
        val wrong = try {
            decrypt(blob, newRecoveryCode()); "FEIL: gal kode ble godtatt"
        } catch (_: BadCodeException) { "gal kode avvist" }
        if (before != after) "FEIL: rundturen endret innholdet"
        else "OK: ${blob.size} byte, $wrong"
    } catch (e: Exception) {
        "FEIL: ${e.message}"
    }
}
