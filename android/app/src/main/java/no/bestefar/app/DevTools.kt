package no.bestefar.app

import android.app.Activity
import android.content.Intent
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.sin
import kotlin.random.Random

/**
 * Utviklerverktøy (musingsUI runde 5). Slå av ved å sette ENABLED=false —
 * da forsvinner Utvikler-menyen fra Avanserte innstillinger.
 */
object DevTools {

    const val ENABLED = true

    /**
     * Semi-tilfeldig serie vektet mot midten, men også mot blinkens areal, så
     * flertallet av «skuddene» legger seg rundt 6–7–8. Lagres som en normal serie.
     */
    fun generateSeries(activity: Activity, shots: Int = 5) {
        val store = Store.get(activity)
        val list = (1..shots).map {
            // Normal(7, 1.6) i poeng gir tyngde rundt 6-8; areal-vekting ivaretas
            // ved at ytre ringer (lavere poeng) er lettere å treffe -> venstre hale.
            var dec = 7.0 + gaussian() * 1.6
            dec = dec.coerceIn(1.0, 10.8)
            val rRel = (10.9 - dec).coerceAtLeast(0.0)   // grov invers av poeng
            val theta = Random.nextDouble(0.0, 2 * Math.PI)
            Shot(decimal = (dec * 10).toInt() / 10.0,
                integer = floor(dec).toInt().coerceIn(0, 10),
                rRel = rRel, theta = theta)
        }
        store.addSeries(SeriesRecord(
            id = Store.newId(), ts = System.currentTimeMillis(),
            weaponId = store.selectedWeapon()?.id, ammoName = "",
            distanceM = store.distanceM,
            position = store.currentPosition, modifier = store.currentModifier,
            shots = list))
        Ui.toast(activity, "Generert serie: sum %.1f".format(list.sumOf { it.decimal }))
    }

    private fun gaussian(): Double {
        // Box–Muller
        val u1 = Random.nextDouble().coerceAtLeast(1e-9)
        val u2 = Random.nextDouble()
        return kotlin.math.sqrt(-2.0 * kotlin.math.ln(u1)) * cos(2 * Math.PI * u2)
    }

    /**
     * «Dummy scan»: fabrikkerer et resultat og åpner ResultActivity, så hele
     * resultat-/stilling-/lagringsflyten kan testes uten kamera. (Bruker ikke
     * Real 1.jpg direkte siden bildet ikke er pakket i appen; fabrikkert
     * treffsett er en enklere og mer stabil test.)
     */
    fun dummyScan(activity: Activity) {
        val n = 5
        val decimals = DoubleArray(n)
        val integers = IntArray(n)
        val rrel = DoubleArray(n)
        val theta = DoubleArray(n)
        for (i in 0 until n) {
            val dec = (6.0 + gaussian() * 1.4).coerceIn(1.0, 10.8)
            decimals[i] = (dec * 10).toInt() / 10.0
            integers[i] = floor(dec).toInt().coerceIn(0, 10)
            rrel[i] = (10.9 - dec).coerceAtLeast(0.0)
            theta[i] = Random.nextDouble(0.0, 2 * Math.PI)
        }
        activity.startActivity(Intent(activity, ResultActivity::class.java)
            .putExtra(ResultActivity.EXTRA_STATUS, BestefarCore.OK)
            .putExtra(ResultActivity.EXTRA_N_HITS, n)
            .putExtra(ResultActivity.EXTRA_CONFIDENCE, 0.9)
            .putExtra(ResultActivity.EXTRA_DECIMALS, decimals)
            .putExtra(ResultActivity.EXTRA_INTEGERS, integers)
            .putExtra(ResultActivity.EXTRA_RREL, rrel)
            .putExtra(ResultActivity.EXTRA_THETA, theta))
    }
}
