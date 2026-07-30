package no.bestefar.app

import android.graphics.BitmapFactory
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton
import java.io.File
import kotlin.math.floor
import kotlin.math.sqrt

/**
 * Resultatkort (musingsUI): skive med treffene markert; poeng i stigende
 * rekkefølge med blyant per poeng; «Ikke lagre» / «OK» under. Runde 4:
 * OCR-finpussing av poeng, innskytingssjekk på sesongens første serie, og
 * varsel om identiske serier. Optikk/klikk-forslag fjernet.
 */
class ResultActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_STATUS = "status"
        const val EXTRA_SUM_DEC = "sum_dec"
        const val EXTRA_SUM_INT = "sum_int"
        const val EXTRA_N_HITS = "n_hits"
        const val EXTRA_CONFIDENCE = "confidence"
        const val EXTRA_DECIMALS = "decimals"
        const val EXTRA_INTEGERS = "integers"
        const val EXTRA_RREL = "r_rel"
        const val EXTRA_THETA = "theta"
        const val EXTRA_IMAGE_PATH = "image_path"
        /** «8. mars 2026    08:38» — god luft mellom dato og tid (musingsUI r7). */
        val DATE_TIME_FMT: java.time.format.DateTimeFormatter =
            java.time.format.DateTimeFormatter.ofPattern("d. MMMM yyyy    HH:mm",
                java.util.Locale("no"))
    }

    private lateinit var store: Store
    private lateinit var content: LinearLayout
    private var record: SeriesRecord? = null
    private var saved = false
    private var imagePath: String? = null
    private var ocrMismatch: List<Double>? = null   // sett -> vis avviks-banner

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = Store.get(this)
        content = Ui.col(this)
        val scroller = Ui.scroll(this, content)
        Ui.applyInsets(scroller)
        setContentView(scroller)
        imagePath = intent.getStringExtra(EXTRA_IMAGE_PATH)

        val status = intent.getIntExtra(EXTRA_STATUS, BestefarCore.ERROR_INTERNAL)
        if (status != BestefarCore.OK) {
            renderRejected(status)
            return
        }

        savedInstanceState?.getString("recordId")?.let { id ->
            record = store.allSeries().firstOrNull { it.id == id }
            if (record != null) { saved = true; render(); return }
        }

        val decimals = intent.getDoubleArrayExtra(EXTRA_DECIMALS) ?: doubleArrayOf()
        val integers = intent.getIntArrayExtra(EXTRA_INTEGERS)
            ?: decimals.map { it.toInt() }.toIntArray()
        val rrel = intent.getDoubleArrayExtra(EXTRA_RREL) ?: DoubleArray(decimals.size)
        val theta = intent.getDoubleArrayExtra(EXTRA_THETA) ?: DoubleArray(decimals.size)
        val shots = decimals.indices.map {
            Shot(decimals[it], integers[it], rrel[it], theta[it])
        }

        resolvePosition { pos, mod ->
            val w = store.selectedWeapon()
            record = SeriesRecord(
                id = Store.newId(),
                ts = System.currentTimeMillis(),
                weaponId = w?.id,
                ammoName = w?.ammoName ?: "",
                distanceM = store.distanceM,
                position = pos,
                modifier = mod,
                shots = shots,
            )
            // Innskytingssjekk før OCR/visning (kan forkaste serien)
            sightInCheck { runOcrThenRender() }
        }
    }

    /** Stilling velges alltid etter scan nå (musingsUI runde 4). */
    private fun resolvePosition(onDone: (Position, PosModifier) -> Unit) {
        Dialogs.positionSheet(this, store, onDone)
    }

    // ---------- Innskyting (musingsUI runde 4) ----------

    private fun sightInCheck(proceed: () -> Unit) {
        val r = record ?: return proceed()
        val shots = r.shots
        if (!Stats.looksMiscalibrated(shots)) return proceed()

        val prior = store.seasonSeriesForWeapon(r.weaponId)
        val today = store.seriesToday().filter { it.weaponId == r.weaponId }

        when {
            // Sesongens (og våpenets) aller første serie
            prior.isEmpty() -> AlertDialog.Builder(this)
                .setTitle(R.string.sightin_title)
                .setMessage(R.string.sightin_body)
                .setPositiveButton(R.string.yes) { _, _ ->
                    Toast.makeText(this, R.string.sightin_registered, Toast.LENGTH_SHORT).show()
                    proceed()
                }
                .setNegativeButton(R.string.no) { _, _ ->
                    Toast.makeText(this, R.string.sightin_discarded, Toast.LENGTH_SHORT).show()
                    finish()
                }
                .setCancelable(false)
                .show()

            // Dagens to første serier med omtrent samme bias
            today.size == 1 && Stats.looksMiscalibrated(today[0].shots) &&
                Stats.similarBias(shots, today[0].shots) -> AlertDialog.Builder(this)
                .setTitle(R.string.sightin_title)
                .setMessage(R.string.sightin_body)
                .setPositiveButton(R.string.yes) { _, _ -> proceed() }
                .setNegativeButton(R.string.no) { _, _ -> askSaveLastTwo(today[0]) { proceed() } }
                .setCancelable(false)
                .show()

            else -> proceed()
        }
    }

    private fun askSaveLastTwo(firstToday: SeriesRecord, proceed: () -> Unit) {
        AlertDialog.Builder(this)
            .setTitle(R.string.sightin_savetwo_title)
            .setPositiveButton(R.string.sightin_save_anyway) { _, _ -> proceed() }
            .setNegativeButton(R.string.sightin_dont_save) { _, _ ->
                // Forkast begge: slett dagens første og ikke lagre denne
                store.deleteSeries(listOf(firstToday.id))
                Toast.makeText(this, R.string.sightin_discarded, Toast.LENGTH_SHORT).show()
                finish()
            }
            .setCancelable(false)
            .show()
    }

    // ---------- OCR-finpussing (musingsUI runde 4) ----------

    private fun runOcrThenRender() {
        val r = record ?: return
        val path = imagePath
        val bmp = if (path != null && File(path).exists())
            try { BitmapFactory.decodeFile(path) } catch (_: Exception) { null } else null
        if (bmp == null) { render(); return }

        OcrVerifier.verify(bmp, r.shots.map { it.decimal }) { result ->
            when (result) {
                is OcrVerifier.Result.Match -> {
                    // Sømløs oppdatering til OCR-poengene (sortert mapping)
                    applyScores(result.scores)
                    if (store.shareDevImagesActive) queueDevImage(r, "ocr_match")
                    render()
                }
                is OcrVerifier.Result.Mismatch -> {
                    ocrMismatch = result.scores
                    render()
                }
                OcrVerifier.Result.Inconclusive -> render()
            }
        }
    }

    /** Erstatt de viste poengene (sortert) med et nytt sett, behold posisjon. */
    private fun applyScores(scores: List<Double>) {
        val r = record ?: return
        val order = r.shots.indices.sortedBy { r.shots[it].decimal }
        val sorted = scores.sorted()
        val newShots = r.shots.toMutableList()
        order.forEachIndexed { rank, idx ->
            val v = sorted[rank]
            newShots[idx] = r.shots[idx].copy(
                decimal = v, integer = floor(v).toInt().coerceIn(0, 10))
        }
        r.shots = newShots
    }

    /** Køer fullskala-bilde + poeng til utvikler (backend-opplasting: TODO). */
    private fun queueDevImage(r: SeriesRecord, tag: String) {
        try {
            val dir = File(filesDir, "dev_uploads").apply { mkdirs() }
            imagePath?.let { p ->
                File(p).copyTo(File(dir, "${r.id}_$tag.jpg"), overwrite = true)
            }
            File(dir, "${r.id}_$tag.json").writeText(
                """{"detected":[${r.shots.joinToString(",") { "%.1f".format(java.util.Locale.US, it.decimal) }}],"tag":"$tag"}""")
        } catch (_: Exception) { /* best effort; backend-opplasting kommer */ }
    }

    private fun renderRejected(status: Int) {
        content.removeAllViews()
        content.addView(Ui.title(this, getString(R.string.app_name)))
        content.addView(Ui.body(this, getString(R.string.result_rejected, status)))
        content.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            text = getString(R.string.result_send_fail)
            layoutParams = Ui.matchWrap(16, this@ResultActivity)
            setOnClickListener {
                Toast.makeText(this@ResultActivity,
                    getString(R.string.summary_queue, 1), Toast.LENGTH_SHORT).show()
                isEnabled = false
            }
        })
        content.addView(MaterialButton(this).apply {
            text = getString(R.string.ok)
            layoutParams = Ui.matchWrap(8, this@ResultActivity)
            setOnClickListener { finish() }
        })
    }

    private fun render() {
        val r = record ?: return
        content.removeAllViews()

        val mainRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        mainRow.addView(TargetView(this).apply {
            hits = r.shots
            layoutParams = LinearLayout.LayoutParams(0,
                Ui.dp(this@ResultActivity, 300), 1f)
        })
        val scoreCol = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            // Poengene midtstilles under «Poeng:» (musingsUI runde 9)
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(Ui.dp(this@ResultActivity, 8), 0, 0, 0)
        }
        scoreCol.addView(TextView(this).apply {
            text = getString(R.string.result_points_label)   // «Poeng:»
            // Samme størrelse som poengene under, i fet (musingsUI runde 8)
            textSize = 19f
            gravity = Gravity.CENTER_HORIZONTAL
            setTypeface(typeface, android.graphics.Typeface.BOLD)
        })
        r.shots.withIndex().sortedBy { it.value.decimal }.forEach { (idx, shot) ->
            // Tettere poeng-rader (musingsUI runde 9): kompakt blyant, liten
            // radhøyde, midtstilt.
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
            }
            row.addView(TextView(this).apply {
                text = "%.1f".format(shot.decimal)
                textSize = 19f
                gravity = Gravity.CENTER
                minWidth = Ui.dp(this@ResultActivity, 40)
            })
            row.addView(ImageButton(this).apply {
                setImageResource(R.drawable.ic_edit)
                background = null
                val p = Ui.dp(this@ResultActivity, 2)
                setPadding(p, p, p, p)
                layoutParams = LinearLayout.LayoutParams(
                    Ui.dp(this@ResultActivity, 26), Ui.dp(this@ResultActivity, 26))
                contentDescription = getString(R.string.result_edit)
                setOnClickListener {
                    Dialogs.shotEdit(this@ResultActivity, shot.decimal) { v ->
                        r.shots = r.shots.mapIndexed { i, s ->
                            if (i == idx) s.copy(decimal = v,
                                integer = floor(v).toInt().coerceAtMost(10)) else s
                        }
                        r.corrected = true
                        if (saved) store.updateSeries(r)
                        render()
                    }
                }
            })
            scoreCol.addView(row, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                topMargin = Ui.dp(this@ResultActivity, 1)
            })
        }
        mainRow.addView(scoreCol)
        content.addView(mainRow)

        // «Poeng:» foran totalen (musingsUI runde 7)
        content.addView(TextView(this).apply {
            text = getString(R.string.result_points_prefix, r.sumDecimal, r.sumInteger)
            textSize = 28f
        })
        // Dato + tid med god luft mellom (musingsUI runde 7)
        content.addView(Ui.body(this,
            java.time.Instant.ofEpochMilli(r.ts).atZone(java.time.ZoneId.systemDefault())
                .format(DATE_TIME_FMT)))
        val w = store.weapons().firstOrNull { it.id == r.weaponId }
        // «Sittende med anlegg» / «... med reim» (musingsUI runde 6)
        val modText = when (r.modifier) {
            PosModifier.ANLEGG -> " med anlegg"
            PosModifier.REIM -> " med reim"
            else -> ""
        }
        content.addView(Ui.hint(this,
            "${r.position.label}$modText · ${r.distanceM} m · " + (w?.shownName ?: "—")))

        // Mitt gjennomsnitt for denne stillingen: brutt ned på stilling (ikke
        // avstand/hjelpemiddel), sesong + totalt, oppdateres løpende (runde 8).
        addAvgBlock(r)

        if (ocrMismatch != null) {
            renderOcrMismatch()
        } else {
            val btnRow = Ui.row(this)
            btnRow.addView(MaterialButton(this, null,
                com.google.android.material.R.attr.borderlessButtonStyle).apply {
                text = getString(R.string.result_discard)
                setOnClickListener { finish() }
            })
            btnRow.addView(android.widget.Space(this),
                LinearLayout.LayoutParams(0, 1, 1f))
            btnRow.addView(MaterialButton(this).apply {
                text = getString(R.string.ok)
                minWidth = Ui.dp(this@ResultActivity, 120)
                setOnClickListener { saveAndFinish() }
            })
            content.addView(btnRow, Ui.matchWrap(12, this))
        }
    }

    /**
     * «Mitt gjennomsnitt for denne stillingen» — sesong + totalt, brutt ned kun
     * på stilling (musingsUI runde 8). Tar med gjeldende serie selv om den ennå
     * ikke er lagret, så snittet stemmer med det som vises.
     */
    private fun addAvgBlock(r: SeriesRecord) {
        val samePos = store.allSeries().filter { it.position == r.position && it.id != r.id } + r
        val seasonNow = Store.seasonKey(r.ts)
        val season = samePos.filter { Store.seasonKey(it.ts) == seasonNow }
        content.addView(TextView(this).apply {
            text = getString(R.string.result_avg_pos_title)
            textSize = 15f
            setTypeface(typeface, android.graphics.Typeface.BOLD)
            setPadding(0, Ui.dp(this@ResultActivity, 8), 0, Ui.dp(this@ResultActivity, 2))
        })
        if (season.isNotEmpty())
            content.addView(Ui.body(this, getString(R.string.result_avg_season,
                season.map { it.sumDecimal }.average())))
        content.addView(Ui.body(this, getString(R.string.result_avg_total,
            samePos.map { it.sumDecimal }.average())))
    }

    /** OCR uenig (> 0.2): vis melding + forkast / lagre med skjermens poeng. */
    private fun renderOcrMismatch() {
        val ocr = ocrMismatch ?: return
        content.addView(Ui.body(this, getString(R.string.ocr_mismatch)))
        content.addView(Ui.hint(this, "Skjermens poeng: " +
            ocr.sorted().joinToString("  ") { "%.1f".format(it) }))
        val btnRow = Ui.row(this)
        btnRow.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = getString(R.string.ocr_discard)
            setOnClickListener { maybeAskDonate { finish() } }
        })
        btnRow.addView(android.widget.Space(this), LinearLayout.LayoutParams(0, 1, 1f))
        btnRow.addView(MaterialButton(this).apply {
            text = getString(R.string.ocr_save_screen)
            setOnClickListener {
                applyScores(ocr)
                ocrMismatch = null
                maybeAskDonate { saveAndFinish() }
            }
        })
        content.addView(btnRow, Ui.matchWrap(12, this))
    }

    /** «Vi vil gjerne bruke dette bildet» — kun hvis dev-deling ikke er aktiv. */
    private fun maybeAskDonate(after: () -> Unit) {
        val r = record
        if (store.shareDevImagesActive || r == null) {
            if (store.shareDevImagesActive) r?.let { queueDevImage(it, "ocr_mismatch") }
            after(); return
        }
        AlertDialog.Builder(this)
            .setTitle(R.string.donate_title)
            .setMessage(R.string.donate_body)
            .setPositiveButton(R.string.donate_accept) { _, _ ->
                store.shareDevImages = "ja"
                queueDevImage(r, "ocr_mismatch")
                after()
            }
            .setNegativeButton(R.string.donate_decline) { _, _ -> after() }
            .setOnCancelListener { after() }
            .show()
    }

    private fun saveAndFinish() {
        val r = record ?: return finish()
        // Varsel kun mot FORRIGE serie (musingsUI runde 5)
        val prev = store.allSeries().filter { it.id != r.id }.maxByOrNull { it.ts }
        if (prev != null && !saved && isIdentical(prev, r)) {
            AlertDialog.Builder(this)
                .setTitle(R.string.dup_title)
                .setMessage(R.string.dup_body)
                .setPositiveButton(R.string.save) { _, _ -> commitAndFinish() }
                .setNegativeButton(R.string.cancel) { _, _ -> finish() }
                .show()
        } else {
            commitAndFinish()
        }
    }

    private fun commitAndFinish() {
        val r = record ?: return finish()
        if (!saved) { store.addSeries(r); saved = true }
        if (r.corrected && !r.sendToFailChannel) {
            Dialogs.failChannelConsent(this) { yes ->
                if (yes) { r.sendToFailChannel = true; store.updateSeries(r) }
                consentFlow()
            }
        } else {
            consentFlow()
        }
    }

    /**
     * Samtykke-rekkefølge (musingsUI runde 6): «Mitt jaktmål» først (etter tre
     * serier), «Bidra til forskning» tidligst to serier senere — aldri begge på
     * samme serie.
     */
    private fun consentFlow() {
        val n = store.currentSeasonSeries().size
        if (!store.jaktmaalPromptSeen && n >= 3) {
            store.jaktmaalPromptSeen = true
            store.jaktmaalPromptCount = n
            Dialogs.jaktmaalDialog(this, store) { finish() }
            return
        }
        Dialogs.maybeResearchConsent(this, store) { finish() }
    }

    private fun isIdentical(a: SeriesRecord, b: SeriesRecord): Boolean {
        if (a.id == b.id) return false
        if (a.position != b.position || a.distanceM != b.distanceM) return false
        if (a.shots.size != b.shots.size || a.shots.isEmpty()) return false
        val sa = a.shots.map { it.decimal }.sorted()
        val sb = b.shots.map { it.decimal }.sorted()
        return sa.indices.all { kotlin.math.abs(sa[it] - sb[it]) < 0.05 }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        if (saved) record?.let { outState.putString("recordId", it.id) }
    }
}
