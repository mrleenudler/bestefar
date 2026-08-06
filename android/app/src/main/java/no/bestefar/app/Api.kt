package no.bestefar.app

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.Handler
import android.os.Looper
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.io.OutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors

/**
 * Tynt HTTP-lag mot backenden (backend_spec.md §5/§10).
 *
 * Bevisst uten nettverksbibliotek: appen snakker med en håndfull endepunkter,
 * og HttpURLConnection holder APK-en liten og byggoppsettet uendret. Vokser
 * behovet (streaming, interceptors, retry-policy) er OkHttp neste steg.
 *
 * Alt her er BLOKKERENDE og skal kalles via [io] — aldri fra UI-tråden.
 *
 * Appen er offline-først: et kall som feiler er ikke en feiltilstand, det er
 * normaltilfellet. Derfor skiller [Resp.retryable] mellom «prøv igjen senere»
 * (nett nede, 429, 5xx) og «dette blir aldri bedre» (4xx), slik at køen kan
 * tømme seg for søppel i stedet for å prøve i evig tid.
 */
object Api {

    private const val TAG = "BestefarApi"
    private const val CONNECT_MS = 15_000
    private const val READ_MS = 30_000

    /** Én tråd: køen skal sendes i rekkefølge, ikke parallelt mot en gratis-tier. */
    private val pool = Executors.newSingleThreadExecutor { r ->
        Thread(r, "bestefar-net").apply { isDaemon = true }
    }
    private val main = Handler(Looper.getMainLooper())

    /** Kjør blokkerende nettverksarbeid utenfor UI-tråden. */
    fun io(block: () -> Unit) {
        pool.execute {
            try { block() } catch (t: Throwable) { Log.e(TAG, "Nettverksjobb feilet", t) }
        }
    }

    /** Hopp tilbake til UI-tråden for å oppdatere visningen. */
    fun ui(block: () -> Unit) { main.post(block) }

    /**
     * Basis-URL. DevTools kan overstyre BuildConfig-verdien i felt (test mot
     * lokal maskin uten å bygge på nytt).
     */
    fun baseUrl(ctx: Context): String {
        val override = Store.get(ctx).apiBaseUrl.trim().trimEnd('/')
        return if (override.isNotEmpty()) override else BuildConfig.API_BASE_URL.trimEnd('/')
    }

    /** `code == 0` betyr at forespørselen aldri kom fram (ingen HTTP-svar). */
    class Resp(val code: Int, val body: String) {
        val ok: Boolean get() = code in 200..299
        val retryable: Boolean get() = code == 0 || code == 408 || code == 429 || code >= 500
    }

    /**
     * Har vi et nett vi har lov til å bruke nå? `validated` er med vilje krevd:
     * et tilkoblet, men portal-fanget wifi gir ellers timeouts i stedet for et
     * ærlig «offline».
     */
    fun online(ctx: Context, wifiOnly: Boolean): Boolean {
        val cm = ctx.getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager
            ?: return false
        val caps = cm.getNetworkCapabilities(cm.activeNetwork ?: return false) ?: return false
        if (!caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)) return false
        if (!caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED)) return false
        if (!wifiOnly) return true
        return caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) ||
            caps.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET)
    }

    // ---------- Forespørsler ----------

    fun postJson(ctx: Context, path: String, body: JSONObject,
                 authRetry: Boolean = true): Resp =
        request(ctx, path, "application/json; charset=utf-8", authRetry = authRetry) { out ->
            out.write(body.toString().toByteArray(Charsets.UTF_8))
        }

    /**
     * Multipart-POST med ett filfelt (`/v1/failed-analyses`). Fila strømmes, så
     * et fullskala-JPEG aldri ligger i minnet i sin helhet.
     */
    fun postMultipart(ctx: Context, path: String, fields: Map<String, String>,
                      fileField: String, file: File, mime: String): Resp {
        val boundary = "----bestefar" + System.nanoTime().toString(16)
        return request(ctx, path, "multipart/form-data; boundary=$boundary") { out ->
            fields.forEach { (k, v) ->
                out.ascii("--$boundary\r\nContent-Disposition: form-data; name=\"$k\"\r\n\r\n")
                out.write(v.toByteArray(Charsets.UTF_8))
                out.ascii("\r\n")
            }
            out.ascii("--$boundary\r\nContent-Disposition: form-data; name=\"$fileField\"; " +
                "filename=\"${file.name}\"\r\nContent-Type: $mime\r\n\r\n")
            FileInputStream(file).use { it.copyTo(out) }
            out.ascii("\r\n--$boundary--\r\n")
        }
    }

    private fun OutputStream.ascii(s: String) = write(s.toByteArray(Charsets.UTF_8))

    /**
     * Rå kropp med valgfri metode — sikkerhetskopien sendes som
     * `application/octet-stream` med metadata i query-strengen (backend_spec
     * §2), ikke som base64 i JSON. Det sparer ~33 % på den største
     * nyttelasten appen har.
     */
    fun send(ctx: Context, method: String, path: String, contentType: String? = null,
             body: ByteArray? = null): Resp =
        request(ctx, path, contentType, method) { out -> body?.let { out.write(it) } }

    /** Som [send], men svaret leses som bytes (nedlasting av kopien). */
    fun download(ctx: Context, path: String, attempt: Int = 0): Pair<Resp, ByteArray?> {
        var conn: HttpURLConnection? = null
        return try {
            conn = open(ctx, path, "GET", null, output = false)
            val code = conn.responseCode
            if (attempt == 0 && shouldRetryAuth(ctx, code, authRetry = true)) {
                conn.disconnect(); conn = null
                return download(ctx, path, 1)
            }
            if (code in 200..299) {
                val bytes = conn.inputStream.use { it.readBytes() }
                Log.d(TAG, "GET $path -> $code (${bytes.size} byte)")
                Resp(code, "") to bytes
            } else {
                val text = conn.errorStream?.bufferedReader(Charsets.UTF_8)
                    ?.use { it.readText() } ?: ""
                Log.d(TAG, "GET $path -> $code")
                Resp(code, text) to null
            }
        } catch (e: Exception) {
            Log.d(TAG, "GET $path feilet: ${e.message}")
            Resp(0, e.message ?: "") to null
        } finally {
            conn?.disconnect()
        }
    }

    private fun open(ctx: Context, path: String, method: String,
                     contentType: String?, output: Boolean): HttpURLConnection =
        (URL(baseUrl(ctx) + path).openConnection() as HttpURLConnection).apply {
            requestMethod = method
            connectTimeout = CONNECT_MS
            readTimeout = READ_MS
            doOutput = output
            contentType?.let { setRequestProperty("Content-Type", it) }
            setRequestProperty("Accept", "application/json")
            setRequestProperty("User-Agent", "Bestefar/${BuildConfig.VERSION_NAME} (Android)")
            // Innlogging (backend_spec §1). Tom uten konto; endepunktene som
            // krever bruker svarer da 401, som er riktig oppførsel og ikke en
            // feil å skjule.
            val token = Auth.accessToken(ctx)
            if (token.isNotEmpty()) setRequestProperty("Authorization", "Bearer $token")
            if (output) setChunkedStreamingMode(0)   // ukjent kroppslengde
        }

    /**
     * Fornyer økten og sier fra om det er verdt å prøve forespørselen på nytt.
     * `authRetry = false` brukes av auth-kallene selv — ellers ville et 401 fra
     * `/v1/auth/refresh` utløst en ny fornyelse, i ring.
     */
    private fun shouldRetryAuth(ctx: Context, code: Int, authRetry: Boolean): Boolean =
        code == 401 && authRetry && Auth.isLoggedIn(ctx) && Auth.refresh(ctx)

    private fun request(ctx: Context, path: String, contentType: String?,
                        method: String = "POST", authRetry: Boolean = true,
                        attempt: Int = 0,
                        writeBody: (OutputStream) -> Unit): Resp {
        var conn: HttpURLConnection? = null
        return try {
            conn = open(ctx, path, method, contentType, output = true)
            conn.outputStream.use(writeBody)
            val code = conn.responseCode
            val stream = if (code in 200..299) conn.inputStream else conn.errorStream
            val text = stream?.bufferedReader(Charsets.UTF_8)?.use { it.readText() } ?: ""
            Log.d(TAG, "$method $path -> $code")
            // Nøyaktig ETT nytt forsøk. Lykkes fornyelsen og vi likevel får 401,
            // er det ikke tokenet som er problemet, og en løkke ville bare
            // brent refresh-tokener.
            if (attempt == 0 && shouldRetryAuth(ctx, code, authRetry)) {
                conn.disconnect(); conn = null
                return request(ctx, path, contentType, method, authRetry, 1, writeBody)
            }
            Resp(code, text)
        } catch (e: Exception) {
            Log.d(TAG, "$method $path feilet: ${e.message}")
            Resp(0, e.message ?: "")
        } finally {
            conn?.disconnect()
        }
    }
}
