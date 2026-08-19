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
        /**
         * OCR leste et annet ANTALL poeng enn kjernen fant treff. Dette er ogsaa
         * `ocr_mismatch`, men retningen er to helt ulike feil:
         *
         * - **Faerre detekterte enn OCR-poeng** ([overDetected] = false): skjulte
         *   treff - to skudd i samme hull. OCR-presedens loeser det: poengene og
         *   summen blir riktige, det skjulte treffet mangler bare et merke.
         * - **Flere detekterte enn OCR-poeng** ([overDetected] = true):
         *   over-deteksjon. Her kan OCR-presedens ikke redde noe, fordi det
         *   overtallige treffet ikke eksisterer - det maa ut av serien.
         *
         * `scores` er OCR-poengene i skjermrekkefoelge.
         */
        data class CountMismatch(
            val scores: List<Double>, val detectedCount: Int,
        ) : Result() {
            val overDetected get() = detectedCount > scores.size
        }
        /** Fant ikke et sammenlignbart sett — la de detekterte stå urørt. */
        object Inconclusive : Result()
    }

    private val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

    private val numberRe = Regex("""\d{1,2}[.,]\d""")

    /**
     * Kjør OCR og sammenlign. `onDone` kalles på hovedtråden.
     *
     * [rotationDegrees] (0/90/180/270) er rotasjonen som gjør [bitmap] opprett.
     * Bitmapen dekodes fra kameraets originalfil og er derfor i
     * sensororientering; ML Kit roterer selv når den får tallet. Sendes 0 for
     * et sideveis bilde, finner den ingen poengliste og svarer `Inconclusive` —
     * altså «fant ikke noe å sammenligne med», ikke «noe gikk galt».
     */
    fun verify(bitmap: Bitmap, rotationDegrees: Int, detected: List<Double>,
               onDone: (Result) -> Unit) {
        if (detected.isEmpty()) { onDone(Result.Inconclusive); return }
        recognizer.process(InputImage.fromBitmap(bitmap, rotationDegrees))
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
        // Fant vi ingen tall, har vi ingen fasit i det hele tatt.
        if (ocr.isEmpty()) return Result.Inconclusive
        // Leste vi FLERE tall enn en serie kan ha skudd, er settet ikke
        // poenglista - da har heuristikken plukket opp tall fra resten av
        // skjermen, og det er ikke noe vi kan sammenligne mot. (Grensa paa de
        // DETEKTERTE treffene haandheves av kalleren, som ogsaa har den naar
        // OCR ikke gir noe svar.)
        if (ocr.size > SeriesRecord.MAX_SHOTS) return Result.Inconclusive
        // Ulikt antall er ikke lenger "kunne ikke sammenlignes": det ER et
        // avvik, og retningen forteller hvilken av de to feilene det er.
        if (ocr.size != detected.size) return Result.CountMismatch(ocr, detected.size)
        // SAMMENLIGNINGEN sorterer (settene skal matche uansett rekkefølge), men
        // `ocr` returneres UROERT — visningen skal vise skjermrekkefølgen.
        val d = detected.sorted()
        val o = ocr.sorted()
        val maxDiff = d.indices.maxOf { abs(d[it] - o[it]) }
        return if (maxDiff <= THRESHOLD) Result.Match(ocr, maxDiff)
        else Result.Mismatch(ocr)
    }
}
