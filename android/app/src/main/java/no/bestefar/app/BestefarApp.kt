package no.bestefar.app

import android.app.Activity
import android.app.Application
import android.os.Bundle
import androidx.appcompat.app.AppCompatDelegate

/** Setter tema (lys/mørk/system) fra lagret valg ved oppstart (musingsUI r4). */
class BestefarApp : Application() {
    override fun onCreate() {
        super.onCreate()
        applyTheme(Store.get(this).themeMode)
        // Opportunistisk tømming av opplastingskøen (backend_spec §6). Alt
        // skjer på en bakgrunnstråd og kan feile stille — appen er offline-
        // først, så et mislykket forsøk er normaltilstanden, ikke en feil.
        Sync.flush(this)
        // Svarte systemlinjer i BÅDE lys og mørk visning (musingsUI runde 12).
        // Registrert her framfor i hver aktivitet: det er tolv aktiviteter, og
        // en ny som glemmer kallet ville fått uleselige systemikoner i lys
        // visning uten at noen oppdager det før i felt.
        registerActivityLifecycleCallbacks(object : ActivityLifecycleCallbacks {
            override fun onActivityCreated(a: Activity, b: Bundle?) = Ui.paintSystemBars(a)
            override fun onActivityStarted(a: Activity) {}
            override fun onActivityResumed(a: Activity) {}
            override fun onActivityPaused(a: Activity) {}
            override fun onActivityStopped(a: Activity) {}
            override fun onActivitySaveInstanceState(a: Activity, b: Bundle) {}
            override fun onActivityDestroyed(a: Activity) {}
        })
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
