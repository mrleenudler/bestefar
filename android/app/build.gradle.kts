// AGP 9+ har INNEBYGD Kotlin — org.jetbrains.kotlin.android skal IKKE brukes.
// kotlinOptions{} er erstattet av kotlin.compilerOptions{}; jvmTarget arver
// android.compileOptions.targetCompatibility, saa den trengs ikke her.
import java.util.Properties

plugins {
    id("com.android.application")
    id("com.google.gms.google-services")
}

// Release-signering: noekler/passord lever i android/keystore.properties +
// android/bestefar-release.keystore - begge GITIGNORERT. Ta backup av begge
// utenfor repoet; mistes keystoren kan appen ikke oppdateres paa enheter.
val keystoreProps = Properties().apply {
    val f = rootProject.file("keystore.properties")
    if (f.exists()) f.inputStream().use { load(it) }
}

// Google-innlogging (backend_spec §1). Web-klient-ID-en (client_type 3) er den
// `aud` backenden verifiserer mot, og den ligger allerede i
// google-services.json - vi leser den derfra i stedet for aa duplisere den inn
// i en gradle.properties som saa kan komme i utakt.
// Android-klienten (client_type 1) brukes IKKE her; Credential Manager vil ha
// SERVER-klient-ID-en. Mangler fila, blir verdien tom og appen skjuler
// Google-knappen framfor aa krasje.
val googleWebClientId: String = run {
    val f = file("google-services.json")
    if (!f.exists()) "" else Regex(
        """"client_id"\s*:\s*"([^"]+)"\s*,\s*"client_type"\s*:\s*3""")
        .find(f.readText())?.groupValues?.get(1) ?: ""
}

android {
    namespace = "no.bestefar.app"
    compileSdk = 36   // CameraX 1.5.x krever compileSdk >= 35

    defaultConfig {
        applicationId = "no.bestefar.app"
        minSdk = 26
        targetSdk = 36
        versionCode = 35
        versionName = "0.35"

        buildConfigField("String", "GOOGLE_WEB_CLIENT_ID", "\"$googleWebClientId\"")

        externalNativeBuild {
            cmake {
                arguments("-DOpenCV_DIR=${project.findProperty("opencvDir") ?: ""}")
                cppFlags("-std=c++17")
            }
        }
        ndk {
            abiFilters += listOf("arm64-v8a")   // utvid ved behov
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    signingConfigs {
        if (keystoreProps.isNotEmpty()) {
            create("release") {
                storeFile = rootProject.file(keystoreProps.getProperty("storeFile"))
                storePassword = keystoreProps.getProperty("storePassword")
                keyAlias = keystoreProps.getProperty("keyAlias")
                keyPassword = keystoreProps.getProperty("keyPassword")
            }
        }
    }

    // BuildConfig er AV som standard fra AGP 8 — vi trenger den til API_BASE_URL.
    buildFeatures {
        buildConfig = true
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            if (keystoreProps.isNotEmpty()) {
                signingConfig = signingConfigs.getByName("release")
            }
            buildConfigField("String", "API_BASE_URL",
                "\"https://bestefar-api.fly.dev\"")
        }
        debug {
            // 10.0.2.2 = vertsmaskinen sett fra emulatoren. Paa fysisk telefon
            // settes adressen i DevTools (Store.apiBaseUrl) i stedet for aa
            // bygge paa nytt. Klartekst-HTTP tillates kun i debug, se
            // src/debug/AndroidManifest.xml.
            buildConfigField("String", "API_BASE_URL", "\"http://10.0.2.2:8000\"")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    // 1.3.x sin libimage_processing_util_jni.so er IKKE 16 kB-alignet
    // (installasjonsfeil paa Android 15+ m/16 kB-kjerne); 1.4.0+ er fikset.
    val camerax = "1.5.1"
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")
    implementation("com.google.android.material:material:1.12.0")
    // On-device OCR: leser skjermens poengtall for aa kontrollere de
    // detekterte treffene (musingsUI runde 4). Bundlet latinsk modell.
    implementation("com.google.mlkit:text-recognition:16.0.1")
    // Opplaasing foran jaktloggen (musingsUI runde 13). Biometri ELLER
    // skjermlaas - androidx.biometric haandterer fallbacken helt ned til v26.
    implementation("androidx.biometric:biometric:1.1.0")
    // Block Store: noekkelen til sikkerhetskopien overlever telefonbytte uten
    // at brukeren maa taste en kode (musingsUI runde 13).
    implementation("com.google.android.gms:play-services-auth-blockstore:16.4.0")
    // Innlogging (v0.17): Credential Manager er Googles gjeldende API. Den
    // gamle Google Sign-In-SDK-en (play-services-auth GoogleSignInClient) er
    // utfaset og skal ikke brukes i nye integrasjoner.
    implementation("androidx.credentials:credentials:1.3.0")
    implementation("androidx.credentials:credentials-play-services-auth:1.3.0")
    implementation("com.google.android.libraries.identity.googleid:googleid:1.1.1")
    // Push (v0.18, backend_spec §11). BoM-en holder Firebase-bibliotekene i
    // takt; vi bruker KUN messaging - ingen analytics, ingen Crashlytics.
    implementation(platform("com.google.firebase:firebase-bom:33.7.0"))
    implementation("com.google.firebase:firebase-messaging")
}
