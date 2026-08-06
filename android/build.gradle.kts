// AGP 9+ har innebygd Kotlin: org.jetbrains.kotlin.android skal ikke deklareres
// (KGP paa classpath gir konflikt med den innebygde stoetten).
plugins {
    id("com.android.application") version "9.2.0" apply false
    // Push (backend_spec §11). Pluginen leser app/google-services.json og
    // genererer ressursene Firebase-biblioteket slaar opp ved oppstart.
    // Mangler fila, FEILER byggingen med vilje - et push-bygg uten
    // prosjektkonfigurasjon ville bare vaert stille oedelagt.
    id("com.google.gms.google-services") version "4.4.2" apply false
}
