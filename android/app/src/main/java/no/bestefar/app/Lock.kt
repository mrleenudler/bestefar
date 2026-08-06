package no.bestefar.app

import android.content.Context
import androidx.biometric.BiometricManager
import androidx.biometric.BiometricPrompt
import androidx.fragment.app.FragmentActivity
import androidx.core.content.ContextCompat

/**
 * Opplåsing foran jaktloggen (musingsUI runde 13).
 *
 * **Vær ærlig om hva dette er.** Det er en DØR foran skjermen, ikke kryptering.
 * Jaktloggen ligger like lesbar på disk som før — den som har rotet telefonen
 * eller kopiert appdataene, ser den uansett. Trusselen låsen faktisk dekker er
 * én ulåst telefon i feil hender, og det er den eneste påstanden UI-teksten
 * skal gjøre.
 *
 * Hvorfor bare jaktloggen: felt vilt, sted og ettersøk er det eneste i appen
 * som er andres sak. Scan-flyten er aldri gatet — den skal virke med hansker,
 * i kulde, på en skytebane der man ikke får av seg votten for å legge fingeren
 * på en sensor.
 */
object Lock {

    /** Biometri ELLER skjermlås. Vi krever ikke fingeravtrykk; PIN er nok. */
    private const val ALLOWED = BiometricManager.Authenticators.BIOMETRIC_WEAK or
        BiometricManager.Authenticators.DEVICE_CREDENTIAL

    /**
     * Frist på fem minutter (musingsUI runde 13). Uten den ville en tur ut og
     * inn i loggen under samme jakt bedt om fingeravtrykk hver gang, og da slår
     * brukeren av hele funksjonen.
     */
    private const val GRACE_MS = 5 * 60_000L

    /** Prosess-levetid med vilje: en ny appstart skal be på nytt. */
    @Volatile private var unlockedAt = 0L

    /** Har enheten i det hele tatt noe å låse opp MED? */
    fun available(ctx: Context): Boolean =
        BiometricManager.from(ctx).canAuthenticate(ALLOWED) ==
            BiometricManager.BIOMETRIC_SUCCESS

    fun enabled(ctx: Context): Boolean = Store.get(ctx).lockHuntLog && available(ctx)

    fun forget() { unlockedAt = 0L }

    /**
     * Kjører [onOk] når brukeren er sluppet inn. Avvist opplåsing gjør
     * INGENTING annet enn å la brukeren stå der de sto — vi lukker ikke appen
     * og kaster dem ikke ut av skjermen de kom fra.
     */
    fun guard(a: FragmentActivity, onOk: () -> Unit) {
        if (!enabled(a)) { onOk(); return }
        if (System.currentTimeMillis() - unlockedAt < GRACE_MS) { onOk(); return }

        val prompt = BiometricPrompt(a, ContextCompat.getMainExecutor(a),
            object : BiometricPrompt.AuthenticationCallback() {
                override fun onAuthenticationSucceeded(r: BiometricPrompt.AuthenticationResult) {
                    unlockedAt = System.currentTimeMillis()
                    onOk()
                }
                override fun onAuthenticationError(code: Int, msg: CharSequence) {
                    // Avbrutt av brukeren skal være stille — de vet hva de
                    // gjorde. Alt annet (ingen maskinvare, låst ute) fortjener
                    // en forklaring, ellers ser appen bare ødelagt ut.
                    if (code != BiometricPrompt.ERROR_USER_CANCELED &&
                        code != BiometricPrompt.ERROR_NEGATIVE_BUTTON &&
                        code != BiometricPrompt.ERROR_CANCELED) {
                        Ui.toast(a, msg.toString())
                    }
                }
            })
        prompt.authenticate(BiometricPrompt.PromptInfo.Builder()
            .setTitle(a.getString(R.string.lock_hunt_prompt_title))
            .setSubtitle(a.getString(R.string.lock_hunt_prompt_body))
            .setAllowedAuthenticators(ALLOWED)
            .build())
    }
}
