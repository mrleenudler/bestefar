package no.bestefar.app

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton

/**
 * Jakt-menyvalg (musingsUI runde 4): to knapper — Registrer jaktskudd og Se
 * registrerte skudd. Scan-knappen vises IKKE her (kun i hovedflaten).
 */
class JaktActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val store = Store.get(this)
        val content = Ui.col(this)
        val scroller = Ui.scroll(this, content)
        Ui.applyInsets(scroller)
        setContentView(scroller)

        content.addView(Ui.title(this, getString(R.string.menu_jakt)))
        content.addView(MaterialButton(this).apply {
            text = getString(R.string.hunt_log_button)
            textSize = 18f
            layoutParams = Ui.matchWrap(8, this@JaktActivity)
                .apply { height = Ui.dp(this@JaktActivity, 64) }
            setOnClickListener {
                Dialogs.maybeHuntConsent(this@JaktActivity, store) {
                    startActivity(Intent(this@JaktActivity, HuntLogActivity::class.java))
                }
            }
        })
        content.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            text = getString(R.string.hunt_view_registered)
            textSize = 18f
            layoutParams = Ui.matchWrap(8, this@JaktActivity)
                .apply { height = Ui.dp(this@JaktActivity, 64) }
            setOnClickListener {
                startActivity(Intent(this@JaktActivity, RegistrerteSkuddActivity::class.java))
            }
        })
    }
}
