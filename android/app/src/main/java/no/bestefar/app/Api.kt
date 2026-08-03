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

    fun postJson(ctx: Context, path: String, body: JSONObject): Resp =
        request(ctx, path, "application/json; charset=utf-8") { out ->
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

    private fun request(ctx: Context, path: String, contentType: String,
                        writeBody: (OutputStream) -> Unit): Resp {
        var conn: HttpURLConnection? = null
        return try {
            conn = (URL(baseUrl(ctx) + path).openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"
                connectTimeout = CONNECT_MS
                readTimeout = READ_MS
                doOutput = true
                setRequestProperty("Content-Type", contentType)
                setRequestProperty("Accept", "application/json")
                setRequestProperty("User-Agent", "Bestefar/${BuildConfig.VERSION_NAME} (Android)")
                // Ukjent kroppslengde (fil kan være stor) -> chunked.
                setChunkedStreamingMode(0)
            }
            conn.outputStream.use(writeBody)
            val code = conn.responseCode
            val stream = if (code in 200..299) conn.inputStream else conn.errorStream
            val text = stream?.bufferedReader(Charsets.UTF_8)?.use { it.readText() } ?: ""
            Log.d(TAG, "POST $path -> $code")
            Resp(code, text)
        } catch (e: Exception) {
            Log.d(TAG, "POST $path feilet: ${e.message}")
            Resp(0, e.message ?: "")
        } finally {
            conn?.disconnect()
        }
    }
}
