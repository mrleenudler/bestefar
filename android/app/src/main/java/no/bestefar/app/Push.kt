package no.bestefar.app

import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.os.Build
import android.util.Log
import com.google.firebase.messaging.FirebaseMessaging
import org.json.JSONObject

/**
 * Push-varsler (backend_spec §11, v0.18).
 *
 * Kjeden er: Firebase gir telefonen en adresse (FCM-tokenet) → vi melder den
 * inn hos backenden → backenden sender varsler dit. Uten mellomleddet skjer
 * ingenting, og det var nettopp mellomleddet som manglet: v0.16 bygget
 * felling-kunngjøringen og v0.17 innloggingen, men `PUT /v1/devices` ble aldri
 * kalt, så backenden hadde ingen adresse å sende til.
 *
 * **Registrering krever konto**, og det er ikke en begrensning vi har funnet
 * på: et varsel er alltid til *noen*. Uten innlogging har vi ingen bruker å
 * knytte adressen til, og da lar vi være — tokenet hentes ikke engang.
 */
object Push {

    private const val TAG = "BestefarPush"

    /**
     * Kanalen må finnes FØR det første varselet, ellers havner varsler fra
     * systemstatusfeltet i en «Diverse»-kanal brukeren ikke kan skru av alene.
     * ID-en står også i manifestet som `default_notification_channel_id`, som
     * er den systemet bruker når appen ligger i bakgrunnen.
     */
    const val CHANNEL = "bestefar_varsler"

    fun ensureChannel(ctx: Context) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val mgr = ctx.getSystemService(NotificationManager::class.java) ?: return
        if (mgr.getNotificationChannel(CHANNEL) != null) return
        mgr.createNotificationChannel(NotificationChannel(CHANNEL,
            ctx.getString(R.string.push_channel_name),
            NotificationManager.IMPORTANCE_DEFAULT).apply {
            description = ctx.getString(R.string.push_channel_desc)
        })
    }

    /**
     * Henter tokenet og melder enheten inn. Idempotent i begge ender —
     * backenden bruker `PUT` nettopp fordi klienten skal kunne kalle dette ved
     * hver oppstart uten å vite om noe har endret seg.
     *
     * Trygg å kalle når som helst: uten innlogging gjør den ingenting.
     */
    fun register(ctx: Context) {
        if (!Auth.isLoggedIn(ctx)) return
        FirebaseMessaging.getInstance().token
            .addOnSuccessListener { token -> onToken(ctx, token) }
            .addOnFailureListener { e ->
                // Ingen Play-tjenester, eller nettet nede ved oppstart. Ikke en
                // feiltilstand i en offline-først app — vi prøver igjen neste
                // gang appen starter.
                Log.d(TAG, "Fikk ikke FCM-token: ${e.message}")
            }
    }

    /** Kalles både fra [register] og fra tjenestens `onNewToken`. */
    fun onToken(ctx: Context, token: String) {
        if (token.isEmpty()) return
        val store = Store.get(ctx)
        store.pushToken = token
        if (!Auth.isLoggedIn(ctx)) return
        Api.io {
            val body = JSONObject()
                .put("push_token", token)
                .put("platform", "android")
                .put("app_version", BuildConfig.VERSION_NAME)
                .put("model", "${Build.MANUFACTURER} ${Build.MODEL}".take(64))
            val resp = Api.send(ctx, "PUT", "/v1/devices",
                "application/json; charset=utf-8",
                body.toString().toByteArray(Charsets.UTF_8))
            Log.d(TAG, "Enhet registrert: ${resp.code}")
        }
    }
}
