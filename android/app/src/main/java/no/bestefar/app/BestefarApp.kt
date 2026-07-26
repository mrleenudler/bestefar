package no.bestefar.app

import android.app.Application
import androidx.appcompat.app.AppCompatDelegate

/** Setter tema (lys/mørk/system) fra lagret valg ved oppstart (musingsUI r4). */
class BestefarApp : Application() {
    override fun onCreate() {
        super.onCreate()
        applyTheme(Store.get(this).themeMode)
    }

    companion object {
        fun applyTheme(mode: String) {
            AppCompatDelegate.setDefaultNightMode(when (mode) {
                "light" -> AppCompatDelegate.MODE_NIGHT_NO
                "dark" -> AppCompatDelegate.MODE_NIGHT_YES
                else -> AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM
            })
        }
    }
}
