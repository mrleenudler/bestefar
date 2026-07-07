package no.bestefar.app

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
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
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Live kamerastroem -> FrameProbe per frame -> auto-capture-trigger ->
 * stillbilde -> analyse -> ResultActivity.
 *
 * Skanner-modell (kravspec §2): bruker holder apparaturen i rammen;
 * appen knipser selv naar stabilitet + kvalitet er oppfylt.
 */
class CaptureActivity : AppCompatActivity() {

    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private val autoCapture = BestefarCore.AutoCapture()
    private val capturing = AtomicBoolean(false)
    private var imageCapture: ImageCapture? = null
    private lateinit var statusText: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_capture)
        statusText = findViewById(R.id.statusText)

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
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build().also { it.setAnalyzer(analysisExecutor, ::onFrame) }

            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA,
                                     preview, imageCapture, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun onFrame(image: ImageProxy) {
        if (capturing.get()) { image.close(); return }
        // Y-planet ER graabildet — det er alt FrameProbe trenger
        val y = image.planes[0]
        val bytes = ByteArray(y.buffer.remaining()).also { y.buffer.get(it) }
        val probe = autoCapture.feed(bytes, image.width, image.height, y.rowStride)
        image.close()

        runOnUiThread {
            statusText.text = when {
                !probe.roiFound -> getString(R.string.status_searching)
                !probe.qualityOk -> getString(R.string.status_quality)
                !probe.stable -> getString(R.string.status_hold_still)
                else -> getString(R.string.status_capturing)
            }
        }
        if (probe.shouldCapture && capturing.compareAndSet(false, true)) {
            takeStillAndAnalyze()
        }
    }

    private fun takeStillAndAnalyze() {
        val ic = imageCapture ?: return
        ic.takePicture(analysisExecutor, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                val ts = System.currentTimeMillis()
                // JPEG -> Bitmap -> RGBA-bytes for kjernen
                val bmp = image.toBitmap()
                image.close()
                val buf = java.nio.ByteBuffer.allocate(bmp.byteCount)
                bmp.copyPixelsToBuffer(buf)
                val result = BestefarCore.analyze(
                    buf.array(), bmp.width, bmp.height, bmp.rowBytes,
                    BestefarCore.FMT_RGBA8, ts)
                startActivity(Intent(this@CaptureActivity, ResultActivity::class.java)
                    .putExtra(ResultActivity.EXTRA_STATUS, result.status)
                    .putExtra(ResultActivity.EXTRA_SUM_DEC, result.sumDecimal)
                    .putExtra(ResultActivity.EXTRA_SUM_INT, result.sumInteger)
                    .putExtra(ResultActivity.EXTRA_N_HITS, result.hits.size)
                    .putExtra(ResultActivity.EXTRA_CONFIDENCE, result.confidence)
                    .putExtra(ResultActivity.EXTRA_DECIMALS,
                              result.hits.map { it.decimal }.toDoubleArray()))
                finish()
            }

            override fun onError(e: ImageCaptureException) {
                capturing.set(false)
            }
        })
    }

    override fun onDestroy() {
        super.onDestroy()
        autoCapture.close()
        analysisExecutor.shutdown()
    }
}
