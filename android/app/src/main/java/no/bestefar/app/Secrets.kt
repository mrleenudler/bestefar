package no.bestefar.app

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import android.util.Log
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * Lite hemmelighetslager for verdier som ikke skal ligge i klartekst i
 * SharedPreferences — innloggingstokenene og nøkkelen til sikkerhetskopien
 * (musingsUI runde 13).
 *
 * **Hvorfor ikke `androidx.security:security-crypto`?** Jetpack Security
 * (EncryptedSharedPreferences) er avviklet av Google, og å ta inn et avviklet
 * bibliotek for seksti linjer kode er en gjeld vi må betale senere uansett.
 * Android Keystore er selve grunnmuren biblioteket sto på, og vi bruker den
 * direkte: én AES-256-GCM-nøkkel som aldri forlater sikkerhetsmaskinvaren,
 * chiffertekst i base64 i en egen prefs-fil.
 *
 * **Egen prefs-fil er et poeng, ikke en detalj.** [Store.exportPrefs] er
 * generisk over *alle* nøklene i `bestefar_ui` — legger vi et token der, havner
 * det i sikkerhetskopien, og en kopi som inneholder innloggingen din er en helt
 * annen sak enn en kopi som inneholder skytterdataene dine.
 *
 * Trusselen dette dekker: en avlest prefs-fil (rootet telefon, adb-backup,
 * filutforsker). Det beskytter ikke mot noen som kjører kode som *appen* — da
 * kan de be Keystore om dekrypteringen selv. Nøkkelen krever bevisst ikke
 * skjermlås/biometri: brukeren skal kunne synke uten å låse opp for hver
 * forespørsel, og [Lock] er den døra som faktisk stenger.
 */
object Secrets {

    private const val TAG = "BestefarSecrets"
    private const val PREFS = "bestefar_secrets"
    private const val ALIAS = "bestefar_secrets_v1"
    private const val IV_LEN = 12
    private const val TAG_BITS = 128

    private fun prefs(ctx: Context) =
        ctx.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    private fun key(): SecretKey {
        val ks = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        (ks.getEntry(ALIAS, null) as? KeyStore.SecretKeyEntry)?.let { return it.secretKey }
        val gen = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
        gen.init(
            KeyGenParameterSpec.Builder(ALIAS,
                KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256)
                .setRandomizedEncryptionRequired(true)
                .build())
        return gen.generateKey()
    }

    /** Tom streng når verdien mangler ELLER ikke lenger kan dekrypteres. */
    fun get(ctx: Context, name: String): String {
        val raw = prefs(ctx).getString(name, null) ?: return ""
        return try {
            val blob = Base64.decode(raw, Base64.NO_WRAP)
            val cipher = Cipher.getInstance("AES/GCM/NoPadding")
            cipher.init(Cipher.DECRYPT_MODE, key(),
                GCMParameterSpec(TAG_BITS, blob, 0, IV_LEN))
            String(cipher.doFinal(blob, IV_LEN, blob.size - IV_LEN), Charsets.UTF_8)
        } catch (e: Exception) {
            // Keystore-nøkkelen kan bli ugyldig (fabrikkgjenoppretting, ny
            // skjermlås på enkelte enheter). Da er verdien tapt for godt —
            // rydd den bort i stedet for å feile på nytt ved hvert oppslag.
            Log.w(TAG, "Kunne ikke dekryptere «$name», forkaster den", e)
            remove(ctx, name)
            ""
        }
    }

    fun put(ctx: Context, name: String, value: String) {
        if (value.isEmpty()) { remove(ctx, name); return }
        try {
            val cipher = Cipher.getInstance("AES/GCM/NoPadding")
            cipher.init(Cipher.ENCRYPT_MODE, key())
            val ct = cipher.doFinal(value.toByteArray(Charsets.UTF_8))
            val blob = cipher.iv + ct
            prefs(ctx).edit()
                .putString(name, Base64.encodeToString(blob, Base64.NO_WRAP)).apply()
        } catch (e: Exception) {
            Log.e(TAG, "Kunne ikke lagre «$name»", e)
        }
    }

    fun remove(ctx: Context, vararg names: String) {
        val e = prefs(ctx).edit()
        names.forEach { e.remove(it) }
        e.apply()
    }
}
