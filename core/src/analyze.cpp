#include "bestefar/analyze.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "hits.h"
#include "outer_circle.h"
#include "perspective.h"
#include "preprocess.h"
#include "rings.h"
#include "scoring.h"
#include "screen.h"

namespace bestefar {
namespace {

// _analyze_core: senter -> ringkalibrering -> perspektivretting ->
// rekalibrering -> treffdeteksjon -> poeng. Koordinater i img_bgr-rammen.
AnalyzeResult analyze_core(const cv::Mat& img_bgr, const Config& cfg) {
    AnalyzeResult out;

    auto oc = detect_outer_circle(img_bgr, cfg);
    if (!oc) {
        out.status = Status::RejectedNoRings;
        out.message = "Ingen edge points funnet";
        return out;
    }

    cv::Mat gray = preprocess::to_gray(img_bgr);

    auto calib_opt = calibrate_and_refine(gray, {oc->cx, oc->cy}, oc->r, cfg);
    if (!calib_opt) {
        out.status = Status::RejectedNoRings;
        out.message = "Bilde forkastet: fant ingen poengringer";
        return out;
    }
    Calibration calib = *calib_opt;
    auto [ok, reason] = validate_calibration(calib, cfg);
    if (!ok) {
        out.status = Status::RejectedInvalidTarget;
        out.message = "Bilde forkastet: ingen gyldig poengskive (" + reason + ")";
        return out;
    }

    // Perspektivretting + rekalibrering i rektifisert ramme
    cv::Mat H;
    if (cfg.persp_rectify_enable) {
        auto Hopt = fit_rectification(calib, cfg);
        if (Hopt) {
            cv::Mat gray_rect = warp_image(gray, *Hopt);
            const cv::Point2d c_rect = transform_point(calib.center, *Hopt);
            double r_rect = 0;
            for (double r : calib.ring_radii_px) r_rect = std::max(r_rect, r);
            auto calib2 = calibrate_and_refine(gray_rect, c_rect, r_rect, cfg,
                                               cfg.ring_recal_iters);
            const bool ok2 = calib2 && validate_calibration(*calib2, cfg).first;
            if (ok2) {
                calib = *calib2;
                gray = gray_rect;
                H = *Hopt;
            }
            // ellers: original kalibrering uten retting (som Python)
        }
    }

    // Treffdeteksjon i analyse-rammen
    auto hit_list = detect_hits(gray, calib, cfg);
    if (cfg.gate_require_hits && hit_list.empty()) {
        out.status = Status::RejectedNoHits;
        out.message = "Bilde forkastet: ingen treff funnet paa skiva";
        return out;
    }

    // Poeng + koordinater tilbake til img_bgr-rammen
    double sum_dec = 0;
    int sum_int = 0;
    double mean_score = 0;
    for (const auto& t : hit_list) {
        const ScorePart sp = score_hit(t.x, t.y, calib, cfg);
        ScoredHit sh;
        sh.r_rel = calib.delta_px > 0 ? sp.distance / calib.delta_px : 0.0;
        sh.theta = sp.theta;
        sh.decimal = sp.decimal;
        sh.integer = sp.integer;
        sh.detect_score = t.score;
        sh.type = t.type;
        cv::Point2d p(t.x, t.y);
        if (!H.empty()) p = transform_point_inverse(p, H);
        sh.x_orig = p.x;
        sh.y_orig = p.y;
        out.hits.push_back(sh);
        sum_dec += sp.decimal;
        sum_int += sp.integer;
        mean_score += t.score;
    }
    std::sort(out.hits.begin(), out.hits.end(),
              [](const ScoredHit& a, const ScoredHit& b) { return a.decimal > b.decimal; });
    out.sum_decimal = std::round(sum_dec * 10.0) / 10.0;
    out.sum_integer = sum_int;

    // Interim-konfidens (OCR-sjekk kommer senere, kravspec §3-merknad)
    out.confidence.n_rings = static_cast<int>(calib.harmonics.size());
    double max_resid = 0;
    for (size_t i = 0; i < calib.ring_values.size(); ++i) {
        const double pred = calib.R10_px + (10.0 - calib.ring_values[i]) * calib.delta_px;
        max_resid = std::max(max_resid, std::abs(calib.ring_radii_px[i] - pred));
    }
    out.confidence.ring_resid_frac = calib.delta_px > 0 ? max_resid / calib.delta_px : 1.0;
    out.confidence.mean_hit_score = hit_list.empty() ? 0.0 : mean_score / hit_list.size();
    const double ring_c = std::clamp(out.confidence.n_rings / 10.0, 0.0, 1.0) *
                          (1.0 - std::clamp(out.confidence.ring_resid_frac /
                                            cfg.gate_max_resid_frac, 0.0, 1.0));
    out.confidence.overall = 0.5 * ring_c + 0.5 * std::clamp(out.confidence.mean_hit_score,
                                                             0.0, 1.0);

    out.status = Status::Ok;
    return out;
}

} // namespace

AnalyzeResult analyze_target(const cv::Mat& img_bgr, const Config& cfg,
                             int64_t timestamp_ms) {
    AnalyzeResult result;
    if (img_bgr.empty()) {
        result.status = Status::ErrorBadInput;
        result.message = "Tomt bilde";
        result.timestamp_ms = timestamp_ms;
        return result;
    }

    try {
        if (cfg.screen_rectify_enable) {
            auto screen = rectify_to_screen(img_bgr, cfg);
            if (screen) {
                AnalyzeResult r = analyze_core(screen->warped, cfg);
                if (r.status == Status::Ok) {
                    // Koordinater tilbake til originalfotoet
                    for (auto& h : r.hits) {
                        const cv::Point2d p = transform_point_inverse(
                            {h.x_orig, h.y_orig}, screen->M);
                        h.x_orig = p.x;
                        h.y_orig = p.y;
                    }
                    r.confidence.screen_used = true;
                    r.timestamp_ms = timestamp_ms;
                    return r;
                }
                if (!cfg.analyze_screen_fallback) {
                    r.timestamp_ms = timestamp_ms;
                    r.confidence.screen_used = true;
                    return r;   // krev gyldig skjermutklipp — ikke fall tilbake
                }
                // ellers: fall gjennom til helbilde-analyse
            }
            // NB (som Python): hvis skjermen IKKE finnes, faller vi ALLTID
            // gjennom til helbilde-analyse — fallback-flagget gjelder kun
            // "skjerm funnet men crop-analysen feilet".
        }
        result = analyze_core(img_bgr, cfg);
        result.timestamp_ms = timestamp_ms;
        return result;
    } catch (const std::exception& e) {
        result.status = Status::ErrorInternal;
        result.message = e.what();
        result.timestamp_ms = timestamp_ms;
        return result;
    }
}

} // namespace bestefar
