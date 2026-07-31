package no.bestefar.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

/**
 * Scan-ramme for capture-skjermen (musings 2026-07, felttest skytebanen):
 * brukerne forstod ikke hvordan telefonen skulle holdes, saa rammen viser
 * HELE skjermlayouten de skal treffe, ikke bare en 4:3-boks:
 *   - sirkel om det hvite skiveomraadet + sirkel om den sorte bullen
 *   - rektangel om poenglista (hoyre side) + rektangel rett under om
 *     oppsummeringen (snitt/S-10/SUM/TOT/Xm/Ym)
 *   - ytre ramme 3x saa tykk som foer, alt matt-gjennomsiktig
 *
 * Geometrien er MAALT paa rektifiserte C-bilder (_probe_frame_geometry*.py,
 * verifisert i Visualiseringer/outputs/scan_frame_mock.png) som andeler av
 * skjermbredde/-hoeyde — hold i synk med _vis_scan_frame_mock.py.
 */
class ScanFrameView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null,
) : View(context, attrs) {

    companion object {
        private const val CX = 0.415f       // ringsenter
        private const val CY = 0.420f
        private const val R_WHITE = 0.304f  // hvit skive-radius (av bredden)
        private const val R_BULL = 0.121f   // sort bull-radius (av bredden)
        private const val TAB_X0 = 0.752f   // skillelinje skive/tabell
        private const val TAB_X1 = 0.990f
        private const val LIST_Y0 = 0.016f  // poengliste topp
        private const val LIST_Y1 = 0.516f  // liste/oppsummering-grense
        private const val SUM_Y1 = 0.824f   // oppsummering bunn

        private const val IDLE_COLOR = 0x66FFFFFFL   // matt-gjennomsiktig hvit
        private const val READY_COLOR = 0xCC4CAF50L  // groenn "Klar!"
    }

    private val outer = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 9f * resources.displayMetrics.density   // 3x gamle 3dp
    }
    private val inner = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2f * resources.displayMetrics.density
    }
    private val rect = RectF()

    /** Groenn "Klar!"-tilstand (vises ETTER at bildet er tatt). */
    var ready: Boolean = false
        set(value) {
            if (field != value) { field = value; invalidate() }
        }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        val color = (if (ready) READY_COLOR else IDLE_COLOR).toInt()
        outer.color = color
        inner.color = color

        val ho = outer.strokeWidth / 2f
        rect.set(ho, ho, w - ho, h - ho)
        canvas.drawRoundRect(rect, 16f, 16f, outer)

        canvas.drawCircle(CX * w, CY * h, R_WHITE * w, inner)
        canvas.drawCircle(CX * w, CY * h, R_BULL * w, inner)

        rect.set(TAB_X0 * w, LIST_Y0 * h, TAB_X1 * w, LIST_Y1 * h)
        canvas.drawRect(rect, inner)
        rect.set(TAB_X0 * w, LIST_Y1 * h, TAB_X1 * w, SUM_Y1 * h)
        canvas.drawRect(rect, inner)
    }
}
