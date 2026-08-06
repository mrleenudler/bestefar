package no.bestefar.app

import android.app.PendingIntent
import android.content.Intent
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import com.google.firebase.messaging.FirebaseMessagingService
import com.google.firebase.messaging.RemoteMessage

/**
 * Mottak av push (backend_spec §11).
 *
 * **Denne tjenesten kalles sjeldnere enn man tror.** Backenden sender en
 * `notification`-blokk, og da tegner Android selv varselet når appen ligger i
 * bakgrunnen — `onMessageReceived` kjøres bare når appen er i forgrunnen.
 * Derfor to ting som må stemme samtidig:
 *
 *  - Manifestet peker på ikon, farge og kanal (`default_notification_*`), som
 *    er det systemet bruker i bakgrunnstilfellet.
 *  - Denne fila bygger det samme varselet for forgrunnstilfellet, ellers ville
 *    en melding som kom mens brukeren så på skjermen forsvinne sporløst.
 */
class PushService : FirebaseMessagingService() {

    /**
     * Firebase roterer tokenet uten forvarsel (ny installasjon, gjenoppretting,
     * app-data tømt). Skjer det uten at vi melder fra, sender backenden videre
     * til en adresse som ikke finnes, og brukeren merker bare at varslene
     * stilner.
     */
    override fun onNewToken(token: String) {
        Push.onToken(this, token)
    }

    override fun onMessageReceived(msg: RemoteMessage) {
        // NB: getterne kalles eksplisitt. Lint (AGP 9) krasjer i sin egen
        // UAST-oppslagskode paa `msg.notification` skrevet som egenskap —
        // «Unexpected failure during lint analysis ... this is a bug in lint».
        // Koden kompilerer fint; det er analysen som faller. Se CLAUDE_CONTEXT.
        @Suppress("UsePropertyAccessSyntax")
        val n = msg.getNotification()
        @Suppress("UsePropertyAccessSyntax")
        val data = msg.getData()
        val title = n?.title ?: data["title"] ?: getString(R.string.app_name)
        val body = n?.body ?: data["body"] ?: return

        Push.ensureChannel(this)

        // Trykk på varselet åpner appen på forsiden. Vi ruter IKKE videre på
        // `kind` ennå: en dyplenke som lander på en tom skjerm er verre enn en
        // som lander på forsiden, og lag-/vennesidene er fortsatt skjelett.
        val intent = Intent(this, MainActivity::class.java)
            .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
        val pending = PendingIntent.getActivity(this, 0, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)

        val varsel = NotificationCompat.Builder(this, Push.CHANNEL)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(title)
            .setContentText(body)
            .setStyle(NotificationCompat.BigTextStyle().bigText(body))
            .setAutoCancel(true)
            .setContentIntent(pending)
            .build()

        // Egen ID per varsel: to gladmeldinger fra to venner skal ikke
        // overskrive hverandre.
        val id = (System.currentTimeMillis() % Int.MAX_VALUE).toInt()
        try {
            NotificationManagerCompat.from(this).notify(id, varsel)
        } catch (_: SecurityException) {
            // Android 13+: brukeren har ikke gitt varseltillatelse. Da skal vi
            // ikke krasje - de har sagt nei, og det er et gyldig svar.
        }
    }
}
