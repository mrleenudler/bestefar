package no.bestefar.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin

/**
 * Skivegjengivelse med plottede skudd (resultatkortet, spec §2).
 * r_rel er i ringavstander fra senter (ring k sin yttergrense ~ r_rel=11-k).
 *
 * THETA-KONVENSJONEN ER Y NEDOVER, ikke matematisk:
 * `theta = atan2(y - cy, x - cx)` med x mot hoeyre og y NEDOVER. Et treff over
 * senter har derfor sin(theta) < 0. Dokumentert i `bestefar_ffi.h` fra
 * 2026-08-22 (issue #12).
 *
 * SAMME AKSEKONVENSJON SOM `x_px/y_px`, MEN IKKE NOEDVENDIGVIS SAMME RAMME.
 * `theta` og `r_rel` regnes i ANALYSERAMMEN, foer kjernens interne skjerm-/
 * perspektivretting; `x_px/y_px` transformeres tilbake gjennom den INVERSE av
 * den rettingen. Rammene faller sammen naar ingen retting ble utfoert, og
 * divergerer generelt naar telefonen var skraatthold. Kjernen advarer derfor
 * eksplisitt: **rekonstruer ikke `x_px/y_px` fra (senter, r_rel, theta)** — de
 * to er uavhengige avlesninger av samme treff, ikke et polar/kartesisk-par i
 * samme ramme. Denne visningen gjoer heller ikke det: den bruker bare
 * `r_rel`/`theta`, i sitt eget skiverom.
 *
 * Fram til v0.29 sto det «antatt matematisk (y opp)» her, og opptegningen
 * speilvendte y for aa kompensere. Antakelsen var feil, og skivevisningen ble
 * speilvendt om den horisontale aksen. Konvensjonen var den gang IKKE
 * dokumentert (headeren sa bare «radianer»), saa fortegnet ble maalt: minste
 * kvadrat paa de fem treffene fra `bestefar_cli Testsett/C1.jpg` ga RMS 1,2 px
 * med y nedover mot 70,5 px med y oppover.
 *
 * **Den lave RMS-en beviser mindre enn den ser ut til.** Den forutsetter at de
 * to rammene faller sammen, og det gjorde de nettopp fordi testbildene er
 * rektifiserte. Metoden generaliserer altsaa IKKE til et skraatthold bilde.
 * Fortegnsfunnet staar — kjernen bekreftet det uavhengig i #12 — men det er
 * konvensjonen som er verifisert, ikke at rammene er én og samme.
 *
 * Skjermens y peker ogsaa nedover, saa riktig avbildning er ren identitet:
 * ingen fortegnsbytte paa noen av aksene.
 */
class TargetView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null,
) : View(context, attrs) {

    var hits: List<Shot> = emptyList()
        set(v) { field = v; invalidate() }

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
    }

    // 11 ringlinjer: inner-tier (0.5), tier..1-er (1..10). Bull (svart) naar
    // ring 7 -> 5 sorte ringer (inner/ytter-tier, 9, 8, 7) og 6 hvite (6..1).
    // maxR = 10: skivekanten faller sammen med 1-er-linja, saa det ikke ligger
    // en ekstra hvit halvring utenfor ytterste ring (musingsUI runde 7).
    private val maxR = 10.0f
    private val bullR = 4.0f
    private val ringRadii = listOf(0.5f) + (1..10).map { it.toFloat() }

    override fun onDraw(canvas: Canvas) {
        val cx = width / 2f
        val cy = height / 2f
        val outer = min(width, height) / 2f - 8f
        val step = outer / maxR

        // Hvit skive + svart blink
        paint.style = Paint.Style.FILL
        paint.color = Color.WHITE
        canvas.drawCircle(cx, cy, outer, paint)
        paint.color = Color.BLACK
        canvas.drawCircle(cx, cy, bullR * step, paint)

        // Ringlinjer: hvite inne i blinken, sorte utenfor
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 1.5f
        for (r in ringRadii) {
            paint.color = if (r <= bullR) Color.WHITE else Color.BLACK
            canvas.drawCircle(cx, cy, r * step, paint)
        }
        // Ytterkant
        paint.color = Color.DKGRAY
        paint.strokeWidth = 3f
        canvas.drawCircle(cx, cy, outer, paint)

        // Skudd
        val holeR = 0.45f * step
        // Et skjult treff har ingen posisjon aa tegne. Da tegner vi det ikke -
        // en prikk i senter ville vaert en paastand vi ikke har dekning for.
        for (h in hits.filter { it.hasPosition }) {
            // Begge fortegn er PLUSS: theta er allerede i y-nedover, og
            // skjermen er y-nedover. Se konvensjonsnotatet paa klassen.
            val x = cx + (h.rRel * cos(h.theta)).toFloat() * step
            val y = cy + (h.rRel * sin(h.theta)).toFloat() * step
            paint.style = Paint.Style.FILL
            paint.color = Color.parseColor("#D32F2F")
            canvas.drawCircle(x, y, holeR, paint)
            paint.style = Paint.Style.STROKE
            paint.strokeWidth = 2f
            paint.color = Color.WHITE
            canvas.drawCircle(x, y, holeR, paint)
        }
    }
}

/**
 * PLASSHOLDER for sidevendt dyrefigur (kompetanseoversikten, spec §5):
 * enkel silhuett (kropp + hode + bein) tonet med forsvarlighetsfargen.
 * Skalerer subtilt med hold; farge bærer hovedbudskapet.
 */
class AnimalView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null,
) : View(context, attrs) {

    var tintColor: Int = Color.GRAY
        set(v) { field = v; invalidate() }

    /** 1.0 = nær; krymper subtilt med hold. */
    var holdScale: Float = 1.0f
        set(v) { field = v.coerceIn(0.6f, 1.0f); invalidate() }

    /** true = speilvendt (vinklings-illusjon inntil ordentlig grafikk). */
    var facingLeft: Boolean = true
        set(v) { field = v; invalidate() }

    var label: String = ""
        set(v) { field = v; invalidate() }

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
        color = Color.WHITE
        isFakeBoldText = true
    }

    override fun onDraw(canvas: Canvas) {
        val w = width.toFloat()
        val h = height.toFloat()
        canvas.save()
        canvas.scale(holdScale, holdScale, w / 2f, h / 2f)
        if (!facingLeft) canvas.scale(-1f, 1f, w / 2f, h / 2f)

        paint.style = Paint.Style.FILL
        paint.color = tintColor

        // Kropp
        val body = RectF(w * 0.22f, h * 0.28f, w * 0.80f, h * 0.62f)
        canvas.drawOval(body, paint)
        // Hals + hode
        canvas.drawOval(RectF(w * 0.10f, h * 0.14f, w * 0.30f, h * 0.42f), paint)
        canvas.drawOval(RectF(w * 0.04f, h * 0.10f, w * 0.20f, h * 0.26f), paint)
        // Bein
        paint.strokeWidth = w * 0.045f
        paint.strokeCap = Paint.Cap.ROUND
        canvas.drawLine(w * 0.30f, h * 0.55f, w * 0.28f, h * 0.88f, paint)
        canvas.drawLine(w * 0.42f, h * 0.58f, w * 0.42f, h * 0.88f, paint)
        canvas.drawLine(w * 0.62f, h * 0.58f, w * 0.62f, h * 0.88f, paint)
        canvas.drawLine(w * 0.72f, h * 0.55f, w * 0.76f, h * 0.88f, paint)
        canvas.restore()

        if (label.isNotEmpty()) {
            textPaint.textSize = h * 0.11f
            textPaint.color = Color.WHITE
            canvas.drawText(label, w / 2f, h * 0.50f, textPaint)
        }
    }
}
