// JNI-bro: Kotlin <-> bestefar_ffi.h. Flat-array-protokoll for resultater
// (enkel, ABI-stabil); parsing skjer i BestefarCore.kt — hold de to i synk.
#include <jni.h>

#include <vector>

#include "bestefar/bestefar_ffi.h"

namespace {

// Resultat-serialisering (holdes i synk med BestefarCore.kt):
// [0]=status [1]=n_hits [2]=sum_decimal [3]=sum_integer [4]=confidence
// [5]=n_rings [6]=timestamp_ms, deretter per treff 7 verdier:
// x, y, r_rel, theta, decimal, integer, detect_score
std::vector<jdouble> serialize(const BfResult& r) {
    std::vector<jdouble> v;
    v.reserve(7 + r.n_hits * 7);
    v.push_back(r.status);
    v.push_back(r.n_hits);
    v.push_back(r.sum_decimal);
    v.push_back(r.sum_integer);
    v.push_back(r.confidence);
    v.push_back(r.n_rings);
    v.push_back(static_cast<jdouble>(r.timestamp_ms));
    for (int i = 0; i < r.n_hits; ++i) {
        const BfHit& h = r.hits[i];
        v.push_back(h.x_px); v.push_back(h.y_px);
        v.push_back(h.r_rel); v.push_back(h.theta);
        v.push_back(h.decimal); v.push_back(h.integer);
        v.push_back(h.detect_score);
    }
    return v;
}

} // namespace

extern "C" {

// analyze(data, w, h, stride, format, tsMs) -> double[] (se serialize)
JNIEXPORT jdoubleArray JNICALL
Java_no_bestefar_app_BestefarCore_nativeAnalyze(JNIEnv* env, jobject /*thiz*/,
                                                jbyteArray data, jint w, jint h,
                                                jint stride, jint format, jlong tsMs) {
    jbyte* buf = env->GetByteArrayElements(data, nullptr);
    BfImage img{reinterpret_cast<const uint8_t*>(buf), w, h, stride, format};
    BfResult res;
    bf_analyze(&img, tsMs, &res);
    env->ReleaseByteArrayElements(data, buf, JNI_ABORT);

    const auto v = serialize(res);
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(v.size()));
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(v.size()), v.data());
    return out;
}

JNIEXPORT jlong JNICALL
Java_no_bestefar_app_BestefarCore_nativeAutoCaptureCreate(JNIEnv*, jobject) {
    return reinterpret_cast<jlong>(bf_autocapture_create(nullptr));
}

JNIEXPORT void JNICALL
Java_no_bestefar_app_BestefarCore_nativeAutoCaptureDestroy(JNIEnv*, jobject, jlong handle) {
    bf_autocapture_destroy(reinterpret_cast<BfAutoCapture*>(handle));
}

// feed(handle, yPlane, w, h, stride) -> double[12]:
// [roi_found, sharpness, clip_lo, clip_hi, coverage, stable, quality_ok,
//  should_capture, quad_x0, quad_y0, roi_width_frac, size_ok]
// (holdes i synk med BestefarCore.kt)
JNIEXPORT jdoubleArray JNICALL
Java_no_bestefar_app_BestefarCore_nativeAutoCaptureFeed(JNIEnv* env, jobject,
                                                        jlong handle, jbyteArray yPlane,
                                                        jint w, jint h, jint stride) {
    jbyte* buf = env->GetByteArrayElements(yPlane, nullptr);
    BfImage img{reinterpret_cast<const uint8_t*>(buf), w, h, stride, BF_FMT_GRAY8};
    BfFrameProbe probe;
    bf_autocapture_feed(reinterpret_cast<BfAutoCapture*>(handle), &img, &probe);
    env->ReleaseByteArrayElements(yPlane, buf, JNI_ABORT);

    const jdouble v[12] = {
        static_cast<jdouble>(probe.roi_found), probe.sharpness,
        probe.clip_lo_frac, probe.clip_hi_frac, probe.coverage,
        static_cast<jdouble>(probe.stable), static_cast<jdouble>(probe.quality_ok),
        static_cast<jdouble>(probe.should_capture), probe.quad[0], probe.quad[1],
        probe.roi_width_frac, static_cast<jdouble>(probe.size_ok)};
    jdoubleArray out = env->NewDoubleArray(12);
    env->SetDoubleArrayRegion(out, 0, 12, v);
    return out;
}

} // extern "C"
