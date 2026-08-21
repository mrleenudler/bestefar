package no.bestefar.app

import android.Manifest
import android.content.ContentValues
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.OrientationEventListener
import android.view.Surface
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Live kamerastroem -> FrameProbe per frame -> auto-capture-trigger ->
 * stillbilde -> lagring (galleri + JSON-sidecar) -> analyse -> ResultActivity.
 *
 * Laast til LANDSKAP: apparatskjermen er 4:3 liggende, saa liggende foto gir
 * best opploesning paa selve skiva. Rammen i overlayet er 4:3.
 *
 * Alle capture-bilder lagres (Pictures/Bestefar + sidecar-JSON i app-mappen)
 * slik at feilanalyser kan hentes ut og kjoeres mot desktop-oraklet.
 */
class CaptureActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "BestefarCapture"

        /**
         * Tidsgrense fra foerste analyse-frame til vi tar bildet uansett.
         *
         * Gatingen er UKALIBRERT (AAP-K1) og bevisst loesnet fordi den var treg
         * paa banen. Den feiler derfor begge veier: et vitrineskap utloeste
         * capture 2026-08-21, mens et skjermbilde av en skive aldri gjorde det.
         * Et bilde som ALDRI tas gir verken analyse eller kalibreringsmateriale
         * — utfallet er identisk med «kjernen feilet», og de to maa kunne
         * skilles (AAP-U14).
         *
         * Verdien er valgt, ikke maalt: 7 s er lenger enn en normal capture
         * bruker paa banen, og kort nok til at brukeren ikke rekker aa gi opp.
         * Den hoerer hjemme i samme kalibrering som tersklene selv.
         */
        private const val CAPTURE_TIMEOUT_MS = 7_000L
    }

    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private val autoCapture = BestefarCore.AutoCapture()
    private val capturing = AtomicBoolean(false)
    private var imageCapture: ImageCapture? = null
    private lateinit var statusText: TextView
    private lateinit var scanFrame: ScanFrameView
    private var loggedFormat = false

    /** Tidsgrensen bor paa hovedtraaden; onFrame kjoerer paa analysetraaden. */
    private val timeoutHandler = android.os.Handler(android.os.Looper.getMainLooper())
    /**
     * Holdes som ETT felt med vilje. `Handler.removeCallbacks` sammenligner
     * Runnable-en med referanselikhet, og `::captureOnTimeout` ville laget en ny
     * instans per kall — nedtellingen hadde da aldri latt seg avbryte.
     */
    private val timeoutRunnable = Runnable { captureOnTimeout() }
    /** Sikrer at nedtellingen startes én gang, ved foerste frame. */
    private val timerArmed = AtomicBoolean(false)
    /**
     * Ble DENNE capturen utloest av tidsgrensen og ikke av gatingen? Leses av
     * [takeStillAndAnalyze] og foelger med til [ResultActivity]; settes foer
     * capturen starter, og bare av den traaden som vant [capturing].
     */
    private var timedOut = false

    // sensorLandscape roterer 180 grader (landskap <-> omvendt landskap) UTEN
    // activity-recreate -> ImageCapture.targetRotation blir staaende paa bind-
    // tidspunktets rotasjon -> stillbildet faar feil rotationDegrees og kjernen
    // ser et opp-ned bilde (felttest skytebanen 2026-07). Loesning: foelg
    // enhetens orientering kontinuerlig (standard CameraX-moenster).
    private val orientationListener by lazy {
        object : OrientationEventListener(this) {
            override fun onOrientationChanged(orientation: Int) {
                if (orientation == ORIENTATION_UNKNOWN) return
                val rotation = when (orientation) {
                    in 45..134 -> Surface.ROTATION_270
                    in 135..224 -> Surface.ROTATION_180
                    in 225..314 -> Surface.ROTATION_90
                    else -> Surface.ROTATION_0
                }
                imageCapture?.targetRotation = rotation
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_capture)
        statusText = findViewById(R.id.statusText)
        scanFrame = findViewById(R.id.scanFrame)
        statusText.text = getString(R.string.status_compose)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1)
        } else {
            startCamera()
        }
    }

    override fun onRequestPermissionsResult(code: Int, perms: Array<out String>,
                                            results: IntArray) {
        super.onRequestPermissionsResult(code, perms, results)
        if (results.firstOrNull() == PackageManager.PERMISSION_GRANTED) startCamera()
        else finish()
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val preview = androidx.camera.core.Preview.Builder().build().also {
                it.setSurfaceProvider(
                    findViewById<PreviewView>(R.id.previewView).surfaceProvider)
            }
            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .build()
            // Eksplisitt YUV: FrameProbe leser plan 0 som graabilde. Ikke stol
            // paa CameraX-defaulten (endret oppfoersel ved 1.3 -> 1.5).
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build().also { it.setAnalyzer(analysisExecutor, ::onFrame) }

            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA,
                                     preview, imageCapture, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun onFrame(image: ImageProxy) {
        if (capturing.get()) { image.close(); return }
        // Bare liggende bilder godkjennes i gatingen (musingsUI runde 4):
        // aktiviteten er sensorLandscape-laast, men portrett-frames droppes
        // eksplisitt som ekstra sikring mot skjeve/opp-ned capture.
        if (image.width < image.height) { image.close(); return }
        // «Scanningen startet» = foerste frame vi faktisk gater paa. Armes her
        // og ikke i onCreate, saa kameraets oppstartstid (og en eventuell
        // tillatelsesdialog) ikke spiser av vinduet. Portrett-frames er
        // allerede droppet over, saa en timeout-capture er liggende som en
        // vanlig capture.
        if (timerArmed.compareAndSet(false, true)) {
            timeoutHandler.postDelayed(timeoutRunnable, CAPTURE_TIMEOUT_MS)
        }
        if (!loggedFormat) {
            loggedFormat = true
            Log.i(TAG, "analyse-frame: ${image.width}x${image.height} " +
                       "format=${image.format} rot=${image.imageInfo.rotationDegrees} " +
                       "pixelStride=${image.planes[0].pixelStride} " +
                       "rowStride=${image.planes[0].rowStride}")
        }
        // Y-planet ER graabildet - det er alt FrameProbe trenger
        val y = image.planes[0]
        val bytes = ByteArray(y.buffer.remaining()).also { y.buffer.get(it) }
        val probe = autoCapture.feed(bytes, image.width, image.height, y.rowStride)
        image.close()
        if (probe == null) return   // activity er under nedrigging

        Log.d(TAG, "probe roi=%b skarp=%.0f klippLo=%.3f klippHi=%.3f dekning=%.2f str=%.2f stabil=%b kval=%b knips=%b"
            .format(probe.roiFound, probe.sharpness, probe.clipLoFrac, probe.clipHiFrac,
                    probe.coverage, probe.screenWidthFrac, probe.stable, probe.qualityOk,
                    probe.shouldCapture))

        // Capture-first (felttest skytebanen 2026-07): bildet tas STILLE i det
        // oeyeblikket kriteriene er oppfylt; deretter spilles brukervennlig UI
        // («Klar!»-groenn ramme 0,4 s -> blits) mens analysen allerede kjoerer.
        if (probe.shouldCapture && capturing.compareAndSet(false, true)) {
            timeoutHandler.removeCallbacks(timeoutRunnable)
            takeStillAndAnalyze()
            runOnUiThread { showReadyThenFlash() }
            return
        }

        runOnUiThread {
            if (capturing.get()) return@runOnUiThread   // «Klar!»-UI eier skjermen
            // Én rolig hovedinstruks; kun reelle kvalitetsblokkere (gjenskinn,
            // lys/fokus) faar egne hint. Ingen groenn ramme foer bildet er tatt.
            statusText.text = when {
                probe.roiFound && !probe.glareOk -> getString(R.string.status_glare)
                probe.roiFound && probe.sizeOk && !probe.qualityOk ->
                    getString(R.string.status_quality)
                else -> getString(R.string.status_compose)
            }
        }
    }

    /**
     * Gatingen slapp ingenting gjennom paa [CAPTURE_TIMEOUT_MS]. Ta gjeldende
     * ramme og kjoer den gjennom analysen likevel.
     *
     * Dette OMGAAR IKKE tilstandsmaskinen i kjernen. `bf_analyze` er en egen
     * FFI-inngang som tar piksler (`bestefar_ffi.h:62`) og har aldri gaatt via
     * `BfAutoCapture` — `takeStillAndAnalyze` kalte den direkte ogsaa foer.
     * Gatingen avgjorde bare NAAR den ble kalt; her er det en andre utloeser
     * til samme kall. Ingen kjerneendring, og ingen probe blir loeyet om.
     *
     * Brukeren skal ikke kunne se forskjell: samme «Klar!»-ramme, samme blits,
     * ingen nedtelling underveis. Forskjellen finnes bare i donasjonen, og der
     * er den poenget — en analyse som LYKKES etter timeout er beviset paa at
     * tersklene i AAP-K1 er for strenge, ikke en feil.
     */
    private fun captureOnTimeout() {
        if (isDestroyed || isFinishing) return
        // Taper vi kapploepet mot gatingen, er bildet allerede tatt.
        if (!capturing.compareAndSet(false, true)) return
        timedOut = true
        Log.i(TAG, "auto-capture timeout etter ${CAPTURE_TIMEOUT_MS} ms - " +
                   "tar gjeldende ramme og analyserer den likevel")
        takeStillAndAnalyze()
        showReadyThenFlash()
    }

    /** Groenn ramme + «Klar!» i 0,4 s, deretter klassisk blits. Bildet er
     *  allerede tatt (stille) naar dette spilles av. */
    private fun showReadyThenFlash() {
        scanFrame.ready = true
        statusText.text = getString(R.string.status_ready)
        scanFrame.postDelayed({ if (!isDestroyed) flash() }, 400)
    }

    /** Klassisk kort hvit blits. */
    private fun flash() {
        val overlay = findViewById<android.view.View>(R.id.flashOverlay)
        overlay.alpha = 0.9f
        overlay.animate().alpha(0f).setDuration(300).start()
    }

    private fun takeStillAndAnalyze() {
        val ic = imageCapture ?: return
        ic.takePicture(analysisExecutor, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                val ts = System.currentTimeMillis()
                // Raa JPEG-bytes (plan 0 for JPEG-format). Disse bytene er
                // kilden til ALT nedenfor: galleri-kopien, pikslene kjernen
                // analyserer, og cache-fila som donasjonen sendes fra. Ingen av
                // de tre komprimeres paa nytt.
                //
                // Kjernen faar ikke JPEG i det hele tatt - den faar dekodede,
                // roterte piksler (se under). Donasjonen er derfor de samme
                // bytene, men i SENSORORIENTERING: `rotationDegrees` er det som
                // skiller den fra det kjernen saa, og foelger med i intenten.
                val jb = image.planes[0].buffer
                val jpeg = ByteArray(jb.remaining()).also { jb.get(it) }
                val rotation = image.imageInfo.rotationDegrees
                image.close()

                val name = "bestefar_" + SimpleDateFormat("yyyyMMdd_HHmmss",
                                                          Locale.US).format(Date(ts))
                // Galleri-lagring er nå brukerens valg (musingsUI runde 10);
                // default på til spørsmålet er stilt etter første scan.
                val galleryUri = if (Store.get(this@CaptureActivity).saveScansToGallery)
                    saveJpegToGallery(jpeg, name) else null

                // Dekod + roter til visningsorientering FOER analyse:
                // toBitmap()/dekoding tar IKKE hensyn til rotationDegrees.
                var bmp = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size)
                if (rotation != 0) {
                    val m = Matrix().apply { postRotate(rotation.toFloat()) }
                    bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
                }
                if (bmp.config != Bitmap.Config.ARGB_8888) {
                    bmp = bmp.copy(Bitmap.Config.ARGB_8888, false)
                }
                val buf = java.nio.ByteBuffer.allocate(bmp.byteCount)
                bmp.copyPixelsToBuffer(buf)
                val result = BestefarCore.analyze(
                    buf.array(), bmp.width, bmp.height, bmp.rowBytes,
                    BestefarCore.FMT_RGBA8, ts)
                saveResultSidecar(name, result, rotation)

                // ORIGINALBYTENE i cache - ikke en omkoding av dem. Fila brukes
                // til to ting: OCR paa skjermens poengtall (musingsUI runde 4)
                // og donasjonen til feilanalyse (backend_spec paragraf 6).
                //
                // Fram til v0.27 laa her `bmp.compress(JPEG, 92)`, altsaa en
                // ANDRE generasjons JPEG av de samme pikslene. For OCR spilte
                // det ingen rolle, men donasjonene er treningsdata for kjernen
                // (AAP-U14), og en detektor finstilt paa omkodede bilder blir
                // finstilt mot artefakter den aldri moeter i drift.
                //
                // Filnavnet baerer kaptur-tidsstempelet. Et fast navn koblet
                // resultat til bilde paa REKKEFOELGE; naa er koblingen en
                // identitet, og en foreldet sti finner ingen fil i stedet for
                // aa finne feil bilde.
                val cacheImg = File(cacheDir, "capture_$ts.jpg")
                try {
                    // Kun én kapring ligger igjen om gangen; originalene er
                    // 6-7 MB. `last_capture.jpg` ryddes for gamle versjoner.
                    cacheDir.listFiles { f ->
                        f.name == "last_capture.jpg" ||
                            (f.name.startsWith("capture_") && f.name.endsWith(".jpg"))
                    }?.forEach { it.delete() }
                    cacheImg.writeBytes(jpeg)
                } catch (e: Exception) { Log.e(TAG, "cache-bilde feilet", e) }

                // Doep om galleri-bildet med fasiten, saa antall treff + sum
                // er synlig rett i galleriet (JSON-sidecaren er vanskelig aa
                // naa uten adb).
                if (galleryUri != null) {
                    val verdict = if (result.isOk)
                        "${result.hits.size}treff_sum${result.sumInteger}"
                    else "avvist${result.status}"
                    try {
                        contentResolver.update(galleryUri, ContentValues().apply {
                            put(MediaStore.Images.Media.DISPLAY_NAME,
                                "${name}_$verdict.jpg")
                        }, null, null)
                    } catch (e: Exception) {
                        Log.e(TAG, "kunne ikke doepe om galleri-bilde", e)
                    }
                }

                startActivity(Intent(this@CaptureActivity, ResultActivity::class.java)
                    .putExtra(ResultActivity.EXTRA_STATUS, result.status)
                    .putExtra(ResultActivity.EXTRA_SUM_DEC, result.sumDecimal)
                    .putExtra(ResultActivity.EXTRA_SUM_INT, result.sumInteger)
                    .putExtra(ResultActivity.EXTRA_N_HITS, result.hits.size)
                    .putExtra(ResultActivity.EXTRA_CONFIDENCE, result.confidence)
                    .putExtra(ResultActivity.EXTRA_DECIMALS,
                              result.hits.map { it.decimal }.toDoubleArray())
                    .putExtra(ResultActivity.EXTRA_INTEGERS,
                              result.hits.map { it.integer }.toIntArray())
                    .putExtra(ResultActivity.EXTRA_RREL,
                              result.hits.map { it.rRel }.toDoubleArray())
                    .putExtra(ResultActivity.EXTRA_THETA,
                              result.hits.map { it.theta }.toDoubleArray())
                    .putExtra(ResultActivity.EXTRA_IMAGE_PATH, cacheImg.absolutePath)
                    // Utloest av tidsgrensen, ikke av gatingen. Foelger med for
                    // aa merke donasjonen; paavirker ingenting i visningen.
                    .putExtra(ResultActivity.EXTRA_TIMED_OUT, timedOut)
                    // Cache-fila er originalen, altsaa i SENSORORIENTERING.
                    // Kjernen fikk pikslene rotert; OCR maa faa den samme
                    // rotasjonen, ellers leser ML Kit et sideveis bilde og
                    // finner ingen poengliste - uten aa melde fra om det.
                    .putExtra(ResultActivity.EXTRA_ROTATION, rotation)
                    .putExtra(ResultActivity.EXTRA_GALLERY_URI, galleryUri?.toString()))
                finish()
            }

            override fun onError(e: ImageCaptureException) {
                Log.e(TAG, "capture feilet", e)
                runOnUiThread {
                    scanFrame.ready = false
                    statusText.text = getString(R.string.status_compose)
                }
                timedOut = false
                // Brukeren scanner videre, saa tidsgrensen skal gjelde paa nytt
                // — ellers ville én mislykket capture skrudd den av for godt.
                timerArmed.set(false)
                capturing.set(false)
            }
        })
    }

    /** Lagre raa JPEG i galleriet (Pictures/Bestefar). Returnerer URI (for
     *  omdoeping med analyse-fasit etterpaa) eller null. */
    private fun saveJpegToGallery(jpeg: ByteArray, name: String): android.net.Uri? {
        return try {
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, "$name.jpg")
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                if (android.os.Build.VERSION.SDK_INT >= 29) {
                    put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/Bestefar")
                }
            }
            val uri = contentResolver.insert(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            if (uri != null) {
                contentResolver.openOutputStream(uri)?.use { it.write(jpeg) }
                Log.i(TAG, "lagret $name.jpg i Pictures/Bestefar")
            }
            uri
        } catch (e: Exception) {
            Log.e(TAG, "kunne ikke lagre bilde", e)
            null
        }
    }

    /** JSON-sidecar med analyseresultatet i app-mappen (adb-hentbar). */
    private fun saveResultSidecar(name: String, r: BestefarCore.AnalyzeResult,
                                  rotation: Int) {
        try {
            val dir = File(getExternalFilesDir(null), "captures").apply { mkdirs() }
            val hitsJson = r.hits.joinToString(",") { h ->
                """{"x":%.1f,"y":%.1f,"r_rel":%.3f,"theta":%.4f,"decimal":%.1f,"integer":%d}"""
                    .format(Locale.US, h.xPx, h.yPx, h.rRel, h.theta, h.decimal, h.integer)
            }
            File(dir, "$name.json").writeText(
                """{"status":%d,"sum_decimal":%.1f,"sum_integer":%d,"confidence":%.3f,"n_rings":%d,"rotation":%d,"hits":[%s]}"""
                    .format(Locale.US, r.status, r.sumDecimal, r.sumInteger,
                            r.confidence, r.nRings, rotation, hitsJson))
            Log.i(TAG, "lagret $name.json")
        } catch (e: Exception) {
            Log.e(TAG, "kunne ikke lagre sidecar", e)
        }
    }

    override fun onStart() {
        super.onStart()
        orientationListener.enable()
    }

    override fun onStop() {
        super.onStop()
        orientationListener.disable()
    }

    override fun onDestroy() {
        super.onDestroy()
        timeoutHandler.removeCallbacks(timeoutRunnable)
        autoCapture.close()
        analysisExecutor.shutdown()
    }
}
