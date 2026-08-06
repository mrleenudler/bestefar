package no.bestefar.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.view.View
import java.time.LocalDate
import java.time.temporal.ChronoUnit

/**
 * Trendgrafen (musingsUI runde 13).
 *
 * Den erstatter «avvik fra snittet», som svarte på feil spørsmål. Brukeren
 * spør ikke *hvor mye over middels er jeg*, men **hvor ligger jeg nå, og går
 * det riktig vei**. Da må tiden være en akse, ikke et tall.
 *
 * - **Førsteaksen** er dato, inntil to jaktår. Sesongskiftet 1. april tegnes
 *   som en loddrett strek, så et fall over sommeren ikke leses som en
 *   plutselig forverring.
 * - **Andreaksen** er poeng per skudd, rullende over 20 skudd (eller dagens
 *   eget snitt når dagen har flere enn 20). Skalaen kommer fra
 *   [Stats.trendAxis] — laveste punkt ligger alltid minst 25 % av aksen under
 *   midten, slik at en flat kurve ser flat ut.
 * - **Dagens eget snitt** tegnes som en svak prikk ved siden av linja. Det er
 *   svaret på «hva med siste økt?»: vi FRAMSKRIVER ingenting. Et beregnet
 *   punkt ville ikke kunne skilles fra et målt, og det er akkurat der
 *   brukeren ser nøyest etter. Prikken viser dagen, linja viser formen.
 */
class TrendView(c: Context) : View(c) {

    private var points: List<Stats.TrendPoint> = emptyList()
    private var lo = 0.0
    private var hi = 1.0

    private val line = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }
    private val dot = Paint(Paint.ANTI_ALIAS_FLAG)
    private val grid = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE }
    private val label = Paint(Paint.ANTI_ALIAS_FLAG)

    private val accent = Ui.themeColor(c, com.google.android.material.R.attr.colorPrimary)
    private val ink = Ui.themeColor(c, android.R.attr.textColorPrimary)

    init {
        line.color = accent
        dot.color = accent
        grid.color = (ink and 0x00FFFFFF) or 0x33000000
        label.color = (ink and 0x00FFFFFF) or 0xAA000000.toInt()
    }

    fun setData(p: List<Stats.TrendPoint>) {
        points = p
        val axis = Stats.trendAxis(p.map { it.value } + p.map { it.dayValue })
        lo = axis.first; hi = axis.second
        contentDescription = if (p.isEmpty()) context.getString(R.string.trend_empty)
        else context.getString(R.string.trend_desc, "%.1f".format(p.last().value))
        invalidate()
    }

    override fun onMeasure(w: Int, h: Int) {
        setMeasuredDimension(MeasureSpec.getSize(w), Ui.dp(context, 180))
    }

    override fun onDraw(canvas: Canvas) {
        val padL = Ui.dp(context, 34).toFloat()
        val padR = Ui.dp(context, 8).toFloat()
        val padT = Ui.dp(context, 10).toFloat()
        val padB = Ui.dp(context, 22).toFloat()
        val w = width - padL - padR
        val h = height - padT - padB
        if (w <= 0 || h <= 0) return

        line.strokeWidth = Ui.dp(context, 2).toFloat()
        grid.strokeWidth = Ui.dp(context, 1).toFloat()
        label.textSize = Ui.dp(context, 10).toFloat()

        // Vannrette hjelpelinjer på hele poeng — de er enheten skytteren
        // tenker i, så aksen skal ikke finne på egne intervaller.
        val first = Math.ceil(lo * 2) / 2.0
        var v = first
        while (v <= hi) {
            val y = padT + h * (1f - ((v - lo) / (hi - lo)).toFloat())
            canvas.drawLine(padL, y, padL + w, y, grid)
            canvas.drawText("%.1f".format(v), Ui.dp(context, 2).toFloat(),
                y + label.textSize / 3f, label)
            v += 0.5
        }

        if (points.isEmpty()) {
            label.textSize = Ui.dp(context, 13).toFloat()
            canvas.drawText(context.getString(R.string.trend_empty),
                padL, padT + h / 2f, label)
            return
        }

        val start = points.first().date
        val end = maxOf(points.last().date, LocalDate.now())
        val span = ChronoUnit.DAYS.between(start, end).coerceAtLeast(1L).toFloat()
        fun x(d: LocalDate) = padL + w * ChronoUnit.DAYS.between(start, d).toFloat() / span
        fun y(value: Double) = padT + h * (1f - ((value - lo) / (hi - lo)).toFloat())

        // Sesongskiftet 1. april: et fall over sommeren er ikke en forverring,
        // det er et nytt jaktår.
        var season = LocalDate.of(start.year, 4, 1)
        if (season.isBefore(start)) season = season.plusYears(1)
        while (!season.isAfter(end)) {
            val sx = x(season)
            canvas.drawLine(sx, padT, sx, padT + h, grid)
            canvas.drawText(Store.seasonLabel(season.year), sx + Ui.dp(context, 3),
                padT + label.textSize, label)
            season = season.plusYears(1)
        }

        // Dagens eget snitt: svak prikk. Sier «dette skjøt du i dag» uten å
        // late som det er trenden.
        dot.alpha = 70
        val r = Ui.dp(context, 2).toFloat()
        points.forEach { canvas.drawCircle(x(it.date), y(it.dayValue), r, dot) }
        dot.alpha = 255

        // Selve trendlinja.
        val path = Path()
        points.forEachIndexed { i, p ->
            val px = x(p.date); val py = y(p.value)
            if (i == 0) path.moveTo(px, py) else path.lineTo(px, py)
        }
        canvas.drawPath(path, line)

        // Siste punkt markeres. Er vinduet ikke fullt ennå, tegnes det åpent —
        // et foreløpig tall skal ikke se like fast ut som et ferdig.
        val last = points.last()
        val lx = x(last.date); val ly = y(last.value)
        val rr = Ui.dp(context, 5).toFloat()
        if (last.partial) {
            line.style = Paint.Style.STROKE
            canvas.drawCircle(lx, ly, rr, line)
        } else {
            canvas.drawCircle(lx, ly, rr, dot)
        }

        canvas.drawText(context.getString(R.string.trend_axis_x),
            padL, (height - Ui.dp(context, 4)).toFloat(), label)
    }
}
