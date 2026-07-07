package no.bestefar.app

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton

/** Foerste iterasjon (kravspec §7): Start -> capture -> resultat -> OK -> Start. */
class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        findViewById<MaterialButton>(R.id.startButton).setOnClickListener {
            startActivity(Intent(this, CaptureActivity::class.java))
        }
    }
}
