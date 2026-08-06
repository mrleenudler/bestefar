package no.bestefar.app

import android.graphics.Color
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Statistikkjernen for UI-et (spec §8): homogen sirkulær normalspredning om
 * siktepunktet antas; P(dødelig) = massen innenfor dødelig radius; alt
 * presenteres i frekvens («9 av 10»), aldri desimaltall (spec §1).
 */
object Stats {

    /**
     * PLASSHOLDER/UKALIBRERT: ringavstand på Kongsberg-skiva i cm per
     * ringenhet (r_rel=1.0). Må settes fra faktisk skivegeometri.
     */
    const val RING_STEP_CM = 2.5

    /**
     * PLASSHOLDER (spec §10.1): dødelig-sone-radius (minste halvakse, cm) per
     * art ved bredside. Skal erstattes av kildebelagt tabell.
     */
    val LETHAL_RADIUS_CM = mapOf(
        Species.ELG to 15.0,
        Species.HJORT to 12.0,
        Species.VILLREIN to 11.0,
        Species.RAADYR to 8.0,
        Species.VILLSVIN to 10.0,
    )

    /**
     * PLASSHOLDER: generell jaktstatistikk som forhåndsinformasjon for
     * stillingsfordeling i felt (øvelsesmotoren, spec §5). Skal erstattes av
     * BH-materiale.
     */
    val HUNT_POSITION_PRIOR = mapOf(
        Position.LIGGENDE to 0.15,
        Position.SITTENDE to 0.30,
        Position.KNESTAENDE to 0.20,
        Position.STAAENDE to 0.35,
    )

    fun lethalRadiusCm(species: Species, angle: Angle): Double? =
        LETHAL_RADIUS_CM[species]?.times(angle.soneFaktor)

    /**
     * σ̂ (cm, 100 m-ekvivalent) fra skudd på tvers av avstander: radiene
     * normaliseres vinkelmessig til 100 m før pooling. E[r²]=2σ² for
     * sirkulær normal. Måles om siktepunktet (senter), ikke MTP (spec §8).
     */
    fun sigmaCmAt100(series: List<SeriesRecord>): Double? {
        val r2 = series.flatMap { s ->
            s.shots.map { sh ->
                val rCm = sh.rRel * RING_STEP_CM * (100.0 / s.distanceM)
                rCm * rCm
            }
        }
        if (r2.isEmpty()) return null
        return sqrt(r2.average() / 2.0)
    }

    fun shotCount(series: List<SeriesRecord>) = series.sumOf { it.shots.size }

    /** P(dødelig): masse innenfor radius R ved avstand d, gitt σ@100m. */
    fun pLethal(sigma100: Double, distanceM: Int, radiusCm: Double): Double {
        val sigmaD = sigma100 * distanceM / 100.0
        if (sigmaD <= 0.0) return 1.0
        return 1.0 - exp(-(radiusCm * radiusCm) / (2.0 * sigmaD * sigmaD))
    }

    /**
     * Grovt spenn på P(dødelig) fra estimeringsusikkerheten i σ
     * (relativ SE ~ 1/√(2n) for radial måling; heuristikk, ikke KI).
     */
    fun pLethalSpan(sigma100: Double, n: Int, distanceM: Int, radiusCm: Double): Pair<Double, Double> {
        val f = 1.0 / sqrt(2.0 * n.coerceAtLeast(1))
        val pLo = pLethal(sigma100 * (1 + f), distanceM, radiusCm)
        val pHi = pLethal(sigma100 * (1 - f).coerceAtLeast(0.1), distanceM, radiusCm)
        return pLo to pHi
    }

    /**
     * Maks forsvarlig hold (m): største d der P(skade) <= valgt rate.
     * P(skade)=exp(-R²/(2σ(d)²)), σ(d)=σ100·d/100 -> løst for d.
     */
    fun maxHoldM(sigma100: Double, radiusCm: Double, rateLimit: Double): Int {
        if (sigma100 <= 0.0) return 999
        val d = radiusCm * 100.0 / (sigma100 * sqrt(2.0 * ln(1.0 / rateLimit)))
        return d.toInt()
    }

    /** Jegerspråk: «~9 av 10» (spec §1). */
    fun freqText(p: Double): String = when {
        p >= 0.995 -> "10 av 10"
        p < 0.05 -> "under 1 av 10"
        else -> "~${(p * 10).roundToInt()} av 10"
    }

    /** Frekvens med spenn ved lite data (spec §5: spenn ved tynt grunnlag). */
    fun freqTextWithSpan(p: Double, pLo: Double, pHi: Double, n: Int): String {
        val lo = (pLo * 10).roundToInt()
        val hi = (pHi * 10).roundToInt()
        return if (n < 30 && lo != hi) "$lo–$hi av 10" else freqText(p)
    }

    /**
     * Farge for felles forsvarlighetskriterium: rød->gul->grønn der gult
     * ligger på brukerens valgte skadeskytingsrate (spec §5).
     */
    fun rateColor(pLethal: Double, rateLimit: Double): Int {
        val threshold = 1.0 - rateLimit
        val hue = if (pLethal >= threshold) {
            60f + 60f * ((pLethal - threshold) / (1.0 - threshold)).toFloat()
        } else {
            60f * (pLethal / threshold).toFloat()
        }
        return Color.HSVToColor(floatArrayOf(hue.coerceIn(0f, 120f), 0.75f, 0.85f))
    }

    /*
     * FJERNET i runde 4 (musingsUI): klikk-forslag krevde optikk-klikkverdi.
     * Optikk-/ammo-/kalkulator-funksjonaliteten er tatt ut for å forenkle
     * brukeropplevelsen (behovet vurderes som marginalt). Beholdes utkommentert
     * som referanse til når/hvis optikk gjeninnføres.
     *
     * fun clickSuggestion(shots, distanceM, clickValueCm100): Pair<Int,Int>? {
     *     ... gjennomsnittlig offset i cm, gated på > ~2σ̂/√n, delt på klikkverdi
     * }
     */

    /**
     * Innskytingssjekk (musingsUI runde 4): en tydelig feilkalibrering viser
     * seg som at gruppas *senter* (bias) ligger langt fra siktepunktet
     * sammenlignet med *spredningen*. Skalafritt forhold, så robust mot at
     * RING_STEP_CM er en plassholder. Returnerer (biasCm, spreadCm).
     */
    fun biasAndSpread(shots: List<Shot>): Pair<Double, Double> {
        if (shots.isEmpty()) return 0.0 to 0.0
        val xs = shots.map { it.rRel * RING_STEP_CM * cos(it.theta) }
        val ys = shots.map { it.rRel * RING_STEP_CM * sin(it.theta) }
        val mx = xs.average(); val my = ys.average()
        val bias = sqrt(mx * mx + my * my)
        val spread = sqrt((xs.indices.sumOf {
            (xs[it] - mx) * (xs[it] - mx) + (ys[it] - my) * (ys[it] - my)
        } / shots.size))
        return bias to spread
    }

    /** Bias-vektor (cm) — brukes til å sammenligne to seriers skjevhet. */
    fun biasVector(shots: List<Shot>): Pair<Double, Double> {
        if (shots.isEmpty()) return 0.0 to 0.0
        return shots.map { it.rRel * RING_STEP_CM * cos(it.theta) }.average() to
            shots.map { it.rRel * RING_STEP_CM * sin(it.theta) }.average()
    }

    /** Tydelig feilkalibrering: senteret ligger godt utenfor egen spredning. */
    fun looksMiscalibrated(shots: List<Shot>): Boolean {
        if (shots.size < 3) return false
        val (bias, spread) = biasAndSpread(shots)
        return bias > 1.5 * spread.coerceAtLeast(0.1) && bias > 2.0 * RING_STEP_CM
    }

    /** To serier har «omtrent samme bias» (samme retning og størrelse). */
    fun similarBias(a: List<Shot>, b: List<Shot>): Boolean {
        val (ax, ay) = biasVector(a)
        val (bx, by) = biasVector(b)
        val d = sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by))
        val mag = (sqrt(ax * ax + ay * ay) + sqrt(bx * bx + by * by)) / 2.0
        return mag > 2.0 * RING_STEP_CM && d < 0.5 * mag
    }

    /** R95 (cm @ 100 m) og MOA — kun «mer statistikk» (spec §1). */
    fun r95Cm(sigma100: Double) = sigma100 * sqrt(-2.0 * ln(0.05))
    fun moa(sigma100: Double): Double {
        val cmPerMoa100 = 2.908
        return 2.0 * sigma100 / cmPerMoa100   // ~diameter av 1σ-gruppe i MOA
    }

    /**
     * Øvelsesmotoren (spec §5): størst underdekning = jaktprior-andel minus
     * treningsandel per hovedstilling. Returnerer (stilling, treningsandel,
     * prior-andel) eller null når gapet er lite eller datagrunnlaget tynt.
     */
    fun practiceSuggestion(series: List<SeriesRecord>): Triple<Position, Double, Double>? {
        val evid = series.filter { it.countsInEvidence }
        if (evid.size < 3) return null
        var best: Triple<Position, Double, Double>? = null
        for (p in Position.hoved) {
            val trained = evid.count { it.position == p }.toDouble() / evid.size
            val prior = HUNT_POSITION_PRIOR[p] ?: 0.0
            val gap = prior - trained
            if (gap > 0.15 && (best == null || gap > best!!.third - best!!.second)) {
                best = Triple(p, trained, prior)
            }
        }
        return best
    }

    // ---------- Trend (musingsUI runde 13) ----------

    /**
     * Vinduet er 20 SKUDD, ikke 20 serier: en serie er 5–10 skudd, så «fem
     * serier» ville betydd 25 for én bruker og 50 for en annen.
     */
    const val TREND_WINDOW = 20

    /**
     * Ett punkt per dag det er skutt.
     *
     * @param value  det som tegnes: rullende snitt over de siste [TREND_WINDOW]
     *               skuddene — eller dagens eget snitt når dagen alene har
     *               flere enn 20 skudd (da er dagen sitt eget vindu).
     * @param dayValue dagens eget snitt, uansett antall. Tegnes som en svak
     *               prikk ved siden av linja, se [TrendView].
     * @param shots  antall skudd i vinduet. Under 20 betyr «foreløpig», og det
     *               vises — et halvfylt vindu skal ikke se ut som et helt.
     */
    class TrendPoint(
        val date: java.time.LocalDate,
        val value: Double,
        val dayValue: Double,
        val dayShots: Int,
        val shots: Int,
    ) {
        val partial: Boolean get() = shots < TREND_WINDOW
    }

    /**
     * Trendkurven for de siste to jaktårene (musingsUI runde 13). Når det
     * tredje jaktåret starter, faller det første ut av seg selv — vi filtrerer
     * på sesongnøkkel, ikke på «730 dager siden», slik at grafen bytter
     * innhold på samme dato som resten av appen bytter sesong.
     */
    fun trendPoints(series: List<SeriesRecord>): List<TrendPoint> {
        val nowSeason = Store.seasonKey(System.currentTimeMillis())
        val kept = series
            .filter { Store.seasonKey(it.ts) >= nowSeason - 1 }
            .sortedBy { it.ts }
        if (kept.isEmpty()) return emptyList()

        val zone = java.time.ZoneId.systemDefault()
        fun dayOf(ts: Long) = java.time.Instant.ofEpochMilli(ts).atZone(zone).toLocalDate()

        // Flat, kronologisk skuddliste — vinduet går på tvers av dager.
        val shots = kept.flatMap { s -> s.shots.map { dayOf(s.ts) to it.decimal } }
        val byDay = shots.groupBy({ it.first }, { it.second })

        val out = ArrayList<TrendPoint>(byDay.size)
        var idx = 0
        byDay.keys.sorted().forEach { day ->
            val own = byDay.getValue(day)
            idx += own.size
            val dayMean = own.average()
            if (own.size > TREND_WINDOW) {
                out.add(TrendPoint(day, dayMean, dayMean, own.size, own.size))
            } else {
                val window = shots.subList((idx - TREND_WINDOW).coerceAtLeast(0), idx)
                out.add(TrendPoint(day, window.map { it.second }.average(),
                    dayMean, own.size, window.size))
            }
        }
        return out
    }

    /**
     * Y-aksen for trendgrafen (musingsUI runde 13): «avstanden fra laveste
     * datapunkt til sentrum skal aldri være mindre enn 25 % av aksen».
     *
     * Regnet ut: med `lo = min − a` og `hi = maks + b` gir kravet
     * `(R + b − a)/2 >= (R + a + b)/4`, altså `a <= (R + b)/3`. I praksis
     * betyr det at det laveste punktet aldri får ligge høyere enn en firedel
     * opp på aksen — luft under kurven er nettopp det som får en flat trend
     * til å se ut som framgang.
     *
     * [minHeight] hindrer at et enkelt punkt havner på gulvet når spennet er 0.
     */
    fun trendAxis(values: List<Double>, minHeight: Double = 1.0): Pair<Double, Double> {
        if (values.isEmpty()) return 0.0 to minHeight
        val m = values.min()
        val M = values.max()
        val r = M - m
        // Litt luft over, og nok totalhøyde til at en flat kurve ikke støyer.
        val b = maxOf(r * 0.15, (minHeight - r) * 0.75, 0.05)
        val a = minOf(maxOf(r * 0.15, (minHeight - r) * 0.25), (r + b) / 3.0)
        return (m - a) to (M + b)
    }

    /** Vinkling i grader for visning (BH-sirkelen kommer, placeholder-tekst). */
    fun angleDeg(a: Angle): Int = when (a) {
        Angle.SIDE -> 90; Angle.SKRAA30 -> 60; Angle.SKRAA60 -> 30
        Angle.FRONT -> 0; Angle.BAK -> 180
    }
}
