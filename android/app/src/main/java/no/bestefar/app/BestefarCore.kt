package no.bestefar.app

/**
 * Kotlin-side av FFI-en. Flat-array-protokollen holdes i synk med
 * jni_bridge.cpp (se serialize() der).
 */
object BestefarCore {
    init {
        System.loadLibrary("bestefar_jni")
    }

    // Statuskoder (= bestefar_ffi.h)
    const val OK = 0
    const val REJECTED_NO_SCREEN = 1
    const val REJECTED_NO_RINGS = 2
    const val REJECTED_INVALID_TARGET = 3
    const val REJECTED_NO_HITS = 4
    const val ERROR_BAD_INPUT = 100
    const val ERROR_INTERNAL = 101

    const val FMT_GRAY8 = 0
    const val FMT_BGR8 = 1
    const val FMT_RGBA8 = 2
    const val FMT_NV21 = 3

    data class Hit(
        val xPx: Double, val yPx: Double,
        val rRel: Double, val theta: Double,
        val decimal: Double, val integer: Int,
        val detectScore: Double,
    )

    data class AnalyzeResult(
        val status: Int,
        val hits: List<Hit>,
        val sumDecimal: Double,
        val sumInteger: Int,
        val confidence: Double,
        val nRings: Int,
        val timestampMs: Long,
    ) {
        val isOk get() = status == OK
    }

    data class FrameProbe(
        val roiFound: Boolean,
        val sharpness: Double,
        val clipLoFrac: Double,
        val clipHiFrac: Double,
        val coverage: Double,
        val stable: Boolean,
        val qualityOk: Boolean,
        val shouldCapture: Boolean,
        val screenWidthFrac: Double,
        val sizeOk: Boolean,
        val bullWidthFrac: Double,
        val glareFrac: Double,
        val glareOk: Boolean,
    )

    fun analyze(data: ByteArray, width: Int, height: Int, stride: Int,
                format: Int, timestampMs: Long): AnalyzeResult {
        val v = nativeAnalyze(data, width, height, stride, format, timestampMs)
        val nHits = v[1].toInt()
        val hits = (0 until nHits).map { i ->
            val o = 7 + i * 7
            Hit(v[o], v[o + 1], v[o + 2], v[o + 3], v[o + 4], v[o + 5].toInt(), v[o + 6])
        }
        return AnalyzeResult(
            status = v[0].toInt(),
            hits = hits,
            sumDecimal = v[2],
            sumInteger = v[3].toInt(),
            confidence = v[4],
            nRings = v[5].toInt(),
            timestampMs = v[6].toLong(),
        )
    }

    /**
     * Auto-capture-tilstandsmaskin. Kall destroy() naar ferdig.
     * feed() og close() er synkronisert: analyse-traaden kan ligge i feed()
     * naar activityen destrueres (tilbake-knapp under scan) — foer krasjet
     * det med IllegalStateException / use-after-free.
     */
    class AutoCapture : AutoCloseable {
        private var handle: Long = nativeAutoCaptureCreate()

        @Synchronized
        fun feed(yPlane: ByteArray, width: Int, height: Int, stride: Int): FrameProbe? {
            if (handle == 0L) return null
            val v = nativeAutoCaptureFeed(handle, yPlane, width, height, stride)
            return FrameProbe(
                roiFound = v[0] > 0.5,
                sharpness = v[1],
                clipLoFrac = v[2],
                clipHiFrac = v[3],
                coverage = v[4],
                stable = v[5] > 0.5,
                qualityOk = v[6] > 0.5,
                shouldCapture = v[7] > 0.5,
                screenWidthFrac = v[10],
                sizeOk = v[11] > 0.5,
                bullWidthFrac = v[12],
                glareFrac = v[13],
                glareOk = v[14] > 0.5,
            )
        }

        @Synchronized
        override fun close() {
            if (handle != 0L) {
                nativeAutoCaptureDestroy(handle)
                handle = 0L
            }
        }
    }

    @JvmStatic private external fun nativeAnalyze(
        data: ByteArray, w: Int, h: Int, stride: Int, format: Int, tsMs: Long): DoubleArray
    @JvmStatic private external fun nativeAutoCaptureCreate(): Long
    @JvmStatic private external fun nativeAutoCaptureDestroy(handle: Long)
    @JvmStatic private external fun nativeAutoCaptureFeed(
        handle: Long, yPlane: ByteArray, w: Int, h: Int, stride: Int): DoubleArray
}
