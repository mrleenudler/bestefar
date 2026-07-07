// AGP 9+ har innebygd Kotlin: org.jetbrains.kotlin.android skal ikke deklareres
// (KGP paa classpath gir konflikt med den innebygde stoetten).
plugins {
    id("com.android.application") version "9.2.0" apply false
}
