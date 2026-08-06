package no.bestefar.app

import android.os.Bundle
import android.os.CountDownTimer
import android.text.InputType
import android.view.ViewGroup
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton

/**
 * Innlogging (backend_spec §1, v0.17).
 *
 * Skjermen har to tilstander og ingen mellomting: enten er du innlogget, eller
 * så er du det ikke. Den ber aldri om noe uoppfordret — du kommer hit fra Min
 * profil fordi du vil noe.
 *
 * Tonen er valgt bevisst: teksten sier hva kontoen *gir deg*, og sier like
 * tydelig at appen virker uten. En jeger som bare vil scanne skiver skal ikke
 * føle at hen har hoppet over noe.
 */
class LoggInnActivity : AppCompatActivity() {

    private lateinit var store: Store
    private lateinit var content: LinearLayout

    /** Nedtelling for «Send ny kode». Serveren bestemmer; vi teller bare ned. */
    private var resendTimer: CountDownTimer? = null
    private var pendingEmail: String = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = Store.get(this)
        content = Ui.col(this)
        val scroller = Ui.scroll(this, content)
        Ui.applyInsets(scroller)
        setContentView(scroller)
        rebuild()
    }

    override fun onDestroy() {
        resendTimer?.cancel()
        super.onDestroy()
    }

    private fun rebuild() {
        resendTimer?.cancel(); resendTimer = null
        content.removeAllViews()
        content.addView(Ui.title(this, getString(R.string.login_title)))
        if (Auth.isLoggedIn(this)) renderLoggedIn() else renderLoggedOut()
    }

    // ---------- Innlogget ----------

    private fun renderLoggedIn() {
        content.addView(Ui.body(this, getString(R.string.login_signed_in_as,
            store.accountName.ifEmpty { getString(R.string.login_unnamed) })))
        if (store.accountPublicId.isNotEmpty()) {
            content.addView(Ui.hint(this,
                getString(R.string.login_public_id, store.accountPublicId)))
        }
        content.addView(Ui.hint(this, getString(R.string.login_what_you_get)))

        content.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            text = getString(R.string.login_logout)
            layoutParams = Ui.matchWrap(24, this@LoggInnActivity)
            setOnClickListener { confirmLogout() }
        })
    }

    /**
     * Utlogging bekreftes, men UTEN advarselsikon: ingenting går tapt. Teksten
     * sier det rett ut, for det er nettopp det brukeren lurer på i det
     * øyeblikket.
     */
    private fun confirmLogout() {
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle(R.string.login_logout)
            .setMessage(R.string.login_logout_confirm)
            .setPositiveButton(R.string.login_logout) { _, _ ->
                Ui.toast(this, R.string.login_working)
                Login.logout(this) {
                    Ui.toast(this, R.string.login_logged_out)
                    rebuild()
                }
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    // ---------- Utlogget ----------

    private fun renderLoggedOut() {
        content.addView(Ui.body(this, getString(R.string.login_intro)))
        content.addView(Ui.hint(this, getString(R.string.login_optional)))

        if (Login.googleConfigured()) {
            content.addView(MaterialButton(this).apply {
                text = getString(R.string.login_google)
                layoutParams = Ui.matchWrap(20, this@LoggInnActivity)
                setOnClickListener { doGoogle(this) }
            })
        }

        content.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            text = getString(R.string.login_email)
            layoutParams = Ui.matchWrap(8, this@LoggInnActivity)
            setOnClickListener { renderEmail() }
        })

        // Apple mangler med vilje: det krever en Apple-utviklerkonto, og den
        // står «på is» hos eier. Backenden har endepunktet klart (§1), så det
        // er én knapp den dagen kontoen finnes.
        content.addView(Ui.hint(this, getString(R.string.login_apple_later)))
    }

    private fun doGoogle(btn: MaterialButton) {
        btn.isEnabled = false
        btn.text = getString(R.string.login_working)
        Login.withGoogle(this) { r ->
            btn.isEnabled = true
            btn.text = getString(R.string.login_google)
            handle(r)
        }
    }

    // ---------- E-postkode ----------

    private fun renderEmail() {
        content.removeAllViews()
        content.addView(Ui.title(this, getString(R.string.login_email)))
        content.addView(Ui.hint(this, getString(R.string.login_email_body)))

        val field = EditText(this).apply {
            hint = getString(R.string.login_email_hint)
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_EMAIL_ADDRESS
            isSingleLine = true
            setText(pendingEmail)
        }
        content.addView(field)

        val send = MaterialButton(this).apply {
            text = getString(R.string.login_email_send)
            layoutParams = Ui.matchWrap(12, this@LoggInnActivity)
        }
        send.setOnClickListener {
            val epost = field.text.toString().trim()
            if (!epost.contains("@") || epost.length < 5) {
                Ui.toast(this, R.string.login_email_invalid); return@setOnClickListener
            }
            pendingEmail = epost
            send.isEnabled = false
            send.text = getString(R.string.login_working)
            Login.startEmail(this, epost) { r, cooldown ->
                send.isEnabled = true
                send.text = getString(R.string.login_email_send)
                if (r is Login.Result.Ok) renderCode(cooldown) else handle(r)
            }
        }
        content.addView(send)

        content.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = getString(R.string.back)
            layoutParams = Ui.matchWrap(8, this@LoggInnActivity)
            setOnClickListener { rebuild() }
        })
    }

    private fun renderCode(cooldown: Int) {
        content.removeAllViews()
        content.addView(Ui.title(this, getString(R.string.login_code_title)))
        content.addView(Ui.body(this, getString(R.string.login_code_body, pendingEmail)))

        val code = EditText(this).apply {
            hint = getString(R.string.login_code_hint)
            inputType = InputType.TYPE_CLASS_NUMBER
            filters = arrayOf(android.text.InputFilter.LengthFilter(6))
            isSingleLine = true
            textSize = 22f
            // Koden kommer i en e-post brukeren nettopp åpnet; autofyll fra
            // SMS finnes ikke her, så feltet får i det minste fokus.
            requestFocus()
        }
        content.addView(code)

        val ok = MaterialButton(this).apply {
            text = getString(R.string.login_code_verify)
            layoutParams = Ui.matchWrap(12, this@LoggInnActivity)
        }
        ok.setOnClickListener {
            val v = code.text.toString().trim()
            if (v.length != 6) { Ui.toast(this, R.string.login_bad_code); return@setOnClickListener }
            ok.isEnabled = false
            ok.text = getString(R.string.login_working)
            Login.verifyEmail(this, pendingEmail, v) { r ->
                ok.isEnabled = true
                ok.text = getString(R.string.login_code_verify)
                handle(r)
            }
        }
        content.addView(ok)

        val resend = MaterialButton(this, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            layoutParams = Ui.matchWrap(4, this@LoggInnActivity)
            setOnClickListener {
                isEnabled = false
                Login.startEmail(this@LoggInnActivity, pendingEmail) { r, c ->
                    if (r is Login.Result.Ok) {
                        Ui.toast(this@LoggInnActivity, R.string.login_code_resent)
                        startResendTimer(this, c)
                    } else { handle(r); startResendTimer(this, c.coerceAtLeast(1)) }
                }
            }
        }
        content.addView(resend)
        startResendTimer(resend, cooldown)

        content.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = getString(R.string.back)
            layoutParams = Ui.matchWrap(8, this@LoggInnActivity)
            setOnClickListener { renderEmail() }
        })
    }

    /**
     * Nedtellingen er bekvemmelighet, ikke sikkerhet — sperrefristen håndheves
     * på serveren (429). Poenget her er at knappen ikke ser trykkbar ut i et
     * minutt der den ikke er det.
     */
    private fun startResendTimer(btn: MaterialButton, seconds: Int) {
        resendTimer?.cancel()
        if (seconds <= 0) {
            btn.isEnabled = true
            btn.text = getString(R.string.login_code_resend)
            return
        }
        btn.isEnabled = false
        resendTimer = object : CountDownTimer(seconds * 1000L, 1000L) {
            override fun onTick(left: Long) {
                btn.text = getString(R.string.login_code_resend_in, (left / 1000).toInt())
            }
            override fun onFinish() {
                btn.isEnabled = true
                btn.text = getString(R.string.login_code_resend)
            }
        }.start()
    }

    // ---------- Felles ----------

    private fun handle(r: Login.Result) {
        when (r) {
            is Login.Result.Ok -> {
                Ui.toast(this, getString(R.string.login_welcome,
                    store.accountName.ifEmpty { getString(R.string.login_unnamed) }))
                rebuild()
                // Nå — og først nå — finnes det varsler å be om tillatelse
                // til. Å spørre ved appstart ville vært et systemvindu uten
                // kontekst, og et «nei» der er nesten umulig å komme tilbake
                // fra: Android viser dialogen bare et par ganger.
                askNotificationsThenRegister()
            }
            // Avbrutt av brukeren: ingen melding. De vet hva de gjorde.
            is Login.Result.Avbrutt -> Unit
            is Login.Result.Feil -> Ui.toast(this, r.melding)
        }
    }

    // ---------- Varsler (backend_spec §11) ----------

    private val askNotifications = registerForActivityResult(
        androidx.activity.result.contract.ActivityResultContracts.RequestPermission()) {
        // Uansett svar: meld inn enheten. Sier brukeren nei, får de ingen
        // varsler nå — men skrur de dem på i systeminnstillingene senere,
        // skal adressen allerede være registrert.
        Push.register(this)
    }

    /**
     * Fra Android 13 er varsler en kjøretidstillatelse. Vi forklarer FØR
     * systemdialogen hva varslene faktisk er: gladmeldinger fra venner og
     * beskjeder fra lagene. Et systemvindu uten kontekst får «nei», og det
     * «nei-et» er nesten permanent.
     */
    private fun askNotificationsThenRegister() {
        if (android.os.Build.VERSION.SDK_INT < 33) { Push.register(this); return }
        val gitt = androidx.core.content.ContextCompat.checkSelfPermission(this,
            android.Manifest.permission.POST_NOTIFICATIONS) ==
            android.content.pm.PackageManager.PERMISSION_GRANTED
        if (gitt) { Push.register(this); return }

        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle(R.string.push_ask_title)
            .setMessage(R.string.push_ask_body)
            .setPositiveButton(R.string.push_ask_yes) { _, _ ->
                askNotifications.launch(android.Manifest.permission.POST_NOTIFICATIONS)
            }
            .setNegativeButton(R.string.push_ask_no) { _, _ -> Push.register(this) }
            .show()
    }
}
