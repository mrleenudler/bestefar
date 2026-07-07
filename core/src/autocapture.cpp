// Auto-capture: FrameProbe (per-frame kvalitet/ROI) + trigger-tilstandsmaskin.
// Kravspec §4: stabilitet og bildekvalitet er LOGISK UAVHENGIGE kriterier.
// Gjenbruker kontrast-ROI-en fra screen-modulen (kravspec §4 "Gjenbruk").
#include <deque>

#include "bestefar/bestefar_ffi.h"
#include "bestefar/config.h"
#include "screen.h"

namespace bestefar {

struct ProbeInternals {
    bool roi_found = false;
    cv::Rect roi_box;
    double sharpness = 0, clip_lo = 0, clip_hi = 0, coverage = 0;
};

// Kjerne-proben: nedskaler -> normaliser -> kontrast-ROI -> metrikker.
static ProbeInternals frame_probe(const cv::Mat& gray_full, const AutoCaptureParams& p,
                                  const Config& cfg) {
    ProbeInternals out;
    const double scale = std::min(static_cast<double>(p.probe_max_side) /
                                  std::max(gray_full.rows, gray_full.cols), 1.0);
    cv::Mat gray;
    if (scale < 1.0)
        cv::resize(gray_full, gray, cv::Size(), scale, scale, cv::INTER_AREA);
    else
        gray = gray_full;

    cv::Mat norm = normalize_stretch(gray, cfg);
    cv::Mat roi = apparatus_roi(norm, cfg);
    if (roi.empty()) return out;
    out.roi_found = true;
    out.roi_box = cv::boundingRect(roi);

    // Skarphet: Laplacian-varians INNE i ROI (fokusmaal)
    cv::Mat lap;
    cv::Laplacian(gray(out.roi_box), lap, CV_32F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev, roi(out.roi_box));
    out.sharpness = stddev[0] * stddev[0];

    // Eksponering: klipp-andeler i ROI (paa U-normalisert graabilde)
    int lo = 0, hi = 0, tot = 0;
    for (int y = out.roi_box.y; y < out.roi_box.y + out.roi_box.height; ++y) {
        const uint8_t* pg = gray.ptr<uint8_t>(y);
        const uint8_t* pr = roi.ptr<uint8_t>(y);
        for (int x = out.roi_box.x; x < out.roi_box.x + out.roi_box.width; ++x) {
            if (!pr[x]) continue;
            ++tot;
            if (pg[x] <= 5) ++lo;
            if (pg[x] >= 250) ++hi;
        }
    }
    if (tot > 0) {
        out.clip_lo = static_cast<double>(lo) / tot;
        out.clip_hi = static_cast<double>(hi) / tot;
    }

    // Dekning: ROI-boksen maa ligge godt innenfor framen (margin)
    const double mx = p.frame_margin_frac * gray.cols;
    const double my = p.frame_margin_frac * gray.rows;
    const cv::Rect inner(static_cast<int>(mx), static_cast<int>(my),
                         static_cast<int>(gray.cols - 2 * mx),
                         static_cast<int>(gray.rows - 2 * my));
    const cv::Rect inter = out.roi_box & inner;
    out.coverage = out.roi_box.area() > 0
                       ? static_cast<double>(inter.area()) / out.roi_box.area()
                       : 0.0;
    // Skaler boks tilbake til frame-koordinater
    if (scale < 1.0) {
        out.roi_box = cv::Rect(static_cast<int>(out.roi_box.x / scale),
                               static_cast<int>(out.roi_box.y / scale),
                               static_cast<int>(out.roi_box.width / scale),
                               static_cast<int>(out.roi_box.height / scale));
    }
    return out;
}

} // namespace bestefar

// ---- C-API for auto-capture -------------------------------------------

struct BfAutoCapture {
    bestefar::AutoCaptureParams params;
    bestefar::Config cfg;
    // Historikk av ROI-bokser for stabilitetskriteriet
    std::deque<cv::Rect> history;
    std::deque<bool> quality_history;
};

extern "C" {

void bf_autocapture_default_params(BfAutoCaptureParams* out) {
    if (!out) return;
    const bestefar::AutoCaptureParams d;
    out->stability_frames = d.stability_frames;
    out->stability_max_move_frac = d.stability_max_move_frac;
    out->min_sharpness = d.min_sharpness;
    out->max_clip_lo_frac = d.max_clip_lo_frac;
    out->max_clip_hi_frac = d.max_clip_hi_frac;
    out->min_coverage = d.min_coverage;
    out->min_roi_width_frac = d.min_roi_width_frac;
    out->frame_margin_frac = d.frame_margin_frac;
    out->probe_max_side = d.probe_max_side;
}

BfAutoCapture* bf_autocapture_create(const BfAutoCaptureParams* p) {
    auto* ac = new BfAutoCapture();
    if (p) {
        ac->params.stability_frames = p->stability_frames;
        ac->params.stability_max_move_frac = p->stability_max_move_frac;
        ac->params.min_sharpness = p->min_sharpness;
        ac->params.max_clip_lo_frac = p->max_clip_lo_frac;
        ac->params.max_clip_hi_frac = p->max_clip_hi_frac;
        ac->params.min_coverage = p->min_coverage;
        ac->params.min_roi_width_frac = p->min_roi_width_frac;
        ac->params.frame_margin_frac = p->frame_margin_frac;
        ac->params.probe_max_side = p->probe_max_side;
    }
    return ac;
}

void bf_autocapture_destroy(BfAutoCapture* ac) { delete ac; }

void bf_autocapture_reset(BfAutoCapture* ac) {
    if (!ac) return;
    ac->history.clear();
    ac->quality_history.clear();
}

} // extern "C"

namespace bestefar {

// Konverter BfImage -> gray cv::Mat (deler buffer der mulig).
cv::Mat bf_image_to_gray(const BfImage* img) {
    if (!img || !img->data || img->width <= 0 || img->height <= 0) return {};
    const cv::Mat wrapped(img->height, img->width,
                          img->format == BF_FMT_GRAY8 ? CV_8UC1
                          : img->format == BF_FMT_RGBA8 ? CV_8UC4
                          : img->format == BF_FMT_BGR8 ? CV_8UC3
                          : CV_8UC1,
                          const_cast<uint8_t*>(img->data),
                          static_cast<size_t>(img->stride));
    cv::Mat gray;
    switch (img->format) {
        case BF_FMT_GRAY8: return wrapped;
        case BF_FMT_BGR8:  cv::cvtColor(wrapped, gray, cv::COLOR_BGR2GRAY); return gray;
        case BF_FMT_RGBA8: cv::cvtColor(wrapped, gray, cv::COLOR_RGBA2GRAY); return gray;
        case BF_FMT_NV21: {
            // NV21: Y-plan foerst — Y ER graabildet
            const cv::Mat y(img->height, img->width, CV_8UC1,
                            const_cast<uint8_t*>(img->data),
                            static_cast<size_t>(img->stride));
            return y;
        }
        default: return {};
    }
}

cv::Mat bf_image_to_bgr(const BfImage* img) {
    if (!img || !img->data || img->width <= 0 || img->height <= 0) return {};
    cv::Mat bgr;
    switch (img->format) {
        case BF_FMT_GRAY8: {
            const cv::Mat g(img->height, img->width, CV_8UC1,
                            const_cast<uint8_t*>(img->data),
                            static_cast<size_t>(img->stride));
            cv::cvtColor(g, bgr, cv::COLOR_GRAY2BGR);
            return bgr;
        }
        case BF_FMT_BGR8: {
            const cv::Mat m(img->height, img->width, CV_8UC3,
                            const_cast<uint8_t*>(img->data),
                            static_cast<size_t>(img->stride));
            return m.clone();
        }
        case BF_FMT_RGBA8: {
            const cv::Mat m(img->height, img->width, CV_8UC4,
                            const_cast<uint8_t*>(img->data),
                            static_cast<size_t>(img->stride));
            cv::cvtColor(m, bgr, cv::COLOR_RGBA2BGR);
            return bgr;
        }
        case BF_FMT_NV21: {
            // Full NV21-buffer: Y-plan + interleavet VU halvopploest
            const cv::Mat yuv(img->height + img->height / 2, img->width, CV_8UC1,
                              const_cast<uint8_t*>(img->data),
                              static_cast<size_t>(img->stride));
            cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV21);
            return bgr;
        }
        default: return {};
    }
}

} // namespace bestefar

extern "C" int32_t bf_autocapture_feed(BfAutoCapture* ac, const BfImage* frame,
                                       BfFrameProbe* out) {
    if (!ac || !frame || !out) return BF_ERROR_BAD_INPUT;
    *out = BfFrameProbe{};
    try {
        const cv::Mat gray = bestefar::bf_image_to_gray(frame);
        if (gray.empty()) return BF_ERROR_BAD_INPUT;

        const auto pr = bestefar::frame_probe(gray, ac->params, ac->cfg);
        out->roi_found = pr.roi_found ? 1 : 0;
        out->sharpness = pr.sharpness;
        out->clip_lo_frac = pr.clip_lo;
        out->clip_hi_frac = pr.clip_hi;
        out->coverage = pr.coverage;
        // quad rapporteres som ROI-boksens hjoerner (TL,TR,BR,BL)
        const cv::Rect& b = pr.roi_box;
        const double q[8] = {
            static_cast<double>(b.x), static_cast<double>(b.y),
            static_cast<double>(b.x + b.width), static_cast<double>(b.y),
            static_cast<double>(b.x + b.width), static_cast<double>(b.y + b.height),
            static_cast<double>(b.x), static_cast<double>(b.y + b.height)};
        for (int i = 0; i < 8; ++i) out->quad[i] = q[i];

        if (!pr.roi_found) {
            ac->history.clear();
            ac->quality_history.clear();
            return BF_OK;
        }

        // KRITERIUM 2: bildekvalitet (uavhengig av stabilitet)
        const auto& p = ac->params;
        const bool quality = pr.sharpness >= p.min_sharpness &&
                             pr.clip_lo <= p.max_clip_lo_frac &&
                             pr.clip_hi <= p.max_clip_hi_frac &&
                             pr.coverage >= p.min_coverage;
        out->quality_ok = quality ? 1 : 0;

        // KRITERIUM 3: stoerrelse — apparatet maa fylle nok av framen.
        // (Liten skjerm gir lav ringavstand i px -> daarlig kalibrering/score.)
        out->roi_width_frac = static_cast<double>(pr.roi_box.width) / gray.cols;
        const bool size_ok = out->roi_width_frac >= p.min_roi_width_frac;
        out->size_ok = size_ok ? 1 : 0;

        // KRITERIUM 1: stabilitet — ROI-boksen i ro over N frames.
        // Holdevinduet krever kvalitet OG stoerrelse gjennom hele vinduet.
        ac->history.push_back(pr.roi_box);
        ac->quality_history.push_back(quality && size_ok);
        while (static_cast<int>(ac->history.size()) > p.stability_frames) {
            ac->history.pop_front();
            ac->quality_history.pop_front();
        }
        bool stable = static_cast<int>(ac->history.size()) >= p.stability_frames;
        if (stable) {
            const cv::Rect& first = ac->history.front();
            const double diag = std::hypot(gray.cols, gray.rows);
            const double max_move = p.stability_max_move_frac * diag;
            for (const auto& r : ac->history) {
                const double d1 = std::hypot(r.x - first.x, r.y - first.y);
                const double d2 = std::hypot((r.x + r.width) - (first.x + first.width),
                                             (r.y + r.height) - (first.y + first.height));
                if (d1 > max_move || d2 > max_move) { stable = false; break; }
            }
        }
        out->stable = stable ? 1 : 0;

        // Trigger: stabil OG kvalitet oppfylt gjennom hele vinduet
        bool all_quality = stable;
        for (const bool qh : ac->quality_history)
            if (!qh) { all_quality = false; break; }
        out->should_capture = (stable && all_quality) ? 1 : 0;
        return BF_OK;
    } catch (...) {
        return BF_ERROR_INTERNAL;
    }
}
