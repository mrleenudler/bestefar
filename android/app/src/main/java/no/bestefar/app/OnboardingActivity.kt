package no.bestefar.app

import android.os.Bundle
import android.view.Gravity
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton

/** Tre korte skjermbilder → rett i første økt (spec §7). */
class OnboardingActivity : AppCompatActivity() {

    private val pages = listOf(
        R.string.onboarding_1_title to R.string.onboarding_1_body,
        R.string.onboarding_2_title to R.string.onboarding_2_body,
        R.string.onboarding_3_title to R.string.onboarding_3_body,
    )
    private var idx = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        render()
    }

    private fun render() {
        val content = Ui.col(this, 32).apply { gravity = Gravity.CENTER }
        val (titleRes, bodyRes) = pages[idx]
        content.addView(TextView(this).apply {
            setText(titleRes); textSize = 26f; gravity = Gravity.CENTER
        })
        content.addView(Ui.vspace(this, 16))
        content.addView(TextView(this).apply {
            setText(bodyRes); textSize = 16f; gravity = Gravity.CENTER
        })
        content.addView(Ui.vspace(this, 32))
        val last = idx == pages.size - 1
        content.addView(MaterialButton(this).apply {
            text = getString(if (last) R.string.onboarding_done else R.string.onboarding_next)
            setOnClickListener {
                if (last) {
                    Store.get(this@OnboardingActivity).onboardingDone = true
                    finish()
                } else {
                    idx++; render()
                }
            }
        })
        setContentView(Ui.scroll(this, content))
    }
}
