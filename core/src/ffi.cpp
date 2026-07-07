// C-FFI-shim rundt analyze_target. Ingen exceptions slipper over grensen.
#include <cstring>

#include "bestefar/analyze.h"
#include "bestefar/bestefar_ffi.h"

namespace bestefar {
cv::Mat bf_image_to_bgr(const BfImage* img);   // autocapture.cpp
}

extern "C" int32_t bf_analyze(const BfImage* image, int64_t timestamp_ms, BfResult* out) {
    if (!out) return BF_ERROR_BAD_INPUT;
    std::memset(out, 0, sizeof(BfResult));
    out->status = BF_ERROR_BAD_INPUT;
    out->timestamp_ms = timestamp_ms;
    if (!image) return out->status;

    try {
        const cv::Mat bgr = bestefar::bf_image_to_bgr(image);
        if (bgr.empty()) return out->status;

        const bestefar::Config cfg;
        const auto res = bestefar::analyze_target(bgr, cfg, timestamp_ms);

        out->status = static_cast<int32_t>(res.status);
        std::strncpy(out->message, res.message.c_str(), sizeof(out->message) - 1);
        out->n_hits = static_cast<int32_t>(std::min(res.hits.size(),
                                                    static_cast<size_t>(BF_MAX_HITS)));
        for (int i = 0; i < out->n_hits; ++i) {
            const auto& h = res.hits[i];
            out->hits[i] = BfHit{h.x_orig, h.y_orig, h.r_rel, h.theta,
                                 h.decimal, h.integer, h.detect_score};
        }
        out->sum_decimal = res.sum_decimal;
        out->sum_integer = res.sum_integer;
        out->confidence = res.confidence.overall;
        out->n_rings = res.confidence.n_rings;
        out->ring_resid_frac = res.confidence.ring_resid_frac;
        out->timestamp_ms = res.timestamp_ms;
        return out->status;
    } catch (const std::exception& e) {
        out->status = BF_ERROR_INTERNAL;
        std::strncpy(out->message, e.what(), sizeof(out->message) - 1);
        return out->status;
    } catch (...) {
        out->status = BF_ERROR_INTERNAL;
        return out->status;
    }
}
