package no.bestefar.app

import android.content.Intent
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton

/**
 * Jakt-menyvalg (musingsUI runde 4/5): to knapper (uten highlight) — Registrer
 * jaktskudd og Se registrerte skudd — pluss «Tilbake» nederst til høyre.
 * Scan-knappen vises IKKE her.
 */
class JaktActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val store = Store.get(this)
        val root = FrameLayout(this)
        Ui.applyInsets(root)
        val content = Ui.col(this)

        content.addView(Ui.title(this, getString(R.string.menu_jakt)))
        content.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
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

        root.addView(Ui.scroll(this, content), ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT)
        // «Tilbake» nederst t.h. -> lukker menyen og går til hovedsiden (runde 6)
        root.addView(MaterialButton(this).apply {
            text = getString(R.string.back)
            setOnClickListener { MainActivity.returnToHome = true; finish() }
        }, FrameLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM or Gravity.END).apply {
            bottomMargin = Ui.dp(this@JaktActivity, 16); rightMargin = Ui.dp(this@JaktActivity, 16)
        })
        setContentView(root)
    }
}
