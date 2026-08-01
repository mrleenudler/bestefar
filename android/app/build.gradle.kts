// AGP 9+ har INNEBYGD Kotlin — org.jetbrains.kotlin.android skal IKKE brukes.
// kotlinOptions{} er erstattet av kotlin.compilerOptions{}; jvmTarget arver
// android.compileOptions.targetCompatibility, saa den trengs ikke her.
import java.util.Properties

plugins {
    id("com.android.application")
}

// Release-signering: noekler/passord lever i android/keystore.properties +
// android/bestefar-release.keystore - begge GITIGNORERT. Ta backup av begge
// utenfor repoet; mistes keystoren kan appen ikke oppdateres paa enheter.
val keystoreProps = Properties().apply {
    val f = rootProject.file("keystore.properties")
    if (f.exists()) f.inputStream().use { load(it) }
}

android {
    namespace = "no.bestefar.app"
    compileSdk = 36   // CameraX 1.5.x krever compileSdk >= 35

    defaultConfig {
        applicationId = "no.bestefar.app"
        minSdk = 26
        targetSdk = 36
        versionCode = 13
        versionName = "0.13"

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

    buildTypes {
        release {
            isMinifyEnabled = false
            if (keystoreProps.isNotEmpty()) {
                signingConfig = signingConfigs.getByName("release")
            }
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
}
