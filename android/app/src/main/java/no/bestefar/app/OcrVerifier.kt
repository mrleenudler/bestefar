package no.bestefar.app

import android.graphics.Bitmap
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlin.math.abs

/**
 * OCR-finpussing av poeng (musingsUI runde 4): leser poengtallene som den
 * elektroniske skiva skriver på skjermen, og kontrollerer de CV-detekterte
 * treffene mot dem.
 *
 * HEURISTIKK / UKALIBRERT: layouten på Kongsberg-skjermen er ikke modellert;
 * vi trekker ut alle desimaltall i [0, 10.9] som kandidat-poeng og matcher dem
 * mot de detekterte poengene. Treffsikkerheten MÅ felttestes. Terskelen 0.2
 * er fra spec.
 */
object OcrVerifier {

    const val THRESHOLD = 0.2

    sealed class Result {
        /**
         * OCR bekrefter (maks avvik <= 0.2). `scores` er OCR-poengene å vise, i
         * SKJERMREKKEFØLGE — altså slik de står i poenglista på apparatet, ikke
         * sortert på verdi (musingsUI runde 10).
         */
        data class Match(val scores: List<Double>, val maxDiff: Double) : Result()
        /** OCR er uenig (> 0.2). `scores` er OCR-poengene i skjermrekkefølge. */
        data class Mismatch(val scores: List<Double>) : Result()
        /** Fant ikke et sammenlignbart sett — la de detekterte stå urørt. */
        object Inconclusive : Result()
    }

    private val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

    private val numberRe = Regex("""\d{1,2}[.,]\d""")

    /** Kjør OCR og sammenlign. `onDone` kalles på hovedtråden. */
    fun verify(bitmap: Bitmap, detected: List<Double>, onDone: (Result) -> Unit) {
        if (detected.isEmpty()) { onDone(Result.Inconclusive); return }
        recognizer.process(InputImage.fromBitmap(bitmap, 0))
            .addOnSuccessListener { text ->
                val nums = numberRe.findAll(text.text)
                    .map { it.value.replace(',', '.').toDouble() }
                    .filter { it in 0.0..10.9 }
                    .toList()
                onDone(compare(detected, nums))
            }
            .addOnFailureListener { onDone(Result.Inconclusive) }
    }

    private fun compare(detected: List<Double>, ocr: List<Double>): Result {
        // Trenger like mange kandidat-poeng som detekterte treff for en ærlig
        // sammenligning; ellers er vi ikke trygge nok til å overstyre.
        if (ocr.size != detected.size) return Result.Inconclusive
        // SAMMENLIGNINGEN sorterer (settene skal matche uansett rekkefølge), men
        // `ocr` returneres UROERT — visningen skal vise skjermrekkefølgen.
        val d = detected.sorted()
        val o = ocr.sorted()
        val maxDiff = d.indices.maxOf { abs(d[it] - o[it]) }
        return if (maxDiff <= THRESHOLD) Result.Match(ocr, maxDiff)
        else Result.Mismatch(ocr)
    }
}
