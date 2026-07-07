#include "overlap.h"

#include <cmath>

#include "circles.h"
#include "hits.h"

namespace bestefar {
namespace {

int odd(double v) {
    int k = static_cast<int>(std::lround(v));
    if (k % 2 == 0) k += 1;
    return std::max(1, k);
}

// _extract_patch med median-fill-padding ved kanter.
cv::Mat extract_patch(const cv::Mat& gray, double cx, double cy, int radius) {
    const int h = gray.rows, w = gray.cols;
    const int size = 2 * radius + 1;
    const int x0 = static_cast<int>(std::lround(cx)) - radius;
    const int y0 = static_cast<int>(std::lround(cy)) - radius;
    const int pad_l = std::max(0, -x0), pad_t = std::max(0, -y0);
    const int pad_r = std::max(0, x0 + size - w), pad_b = std::max(0, y0 + size - h);
    const int x0c = std::max(0, x0), y0c = std::max(0, y0);
    const int x1c = std::min(w, x0 + size), y1c = std::min(h, y0 + size);
    cv::Mat crop = gray(cv::Rect(x0c, y0c, x1c - x0c, y1c - y0c));
    if (pad_l || pad_t || pad_r || pad_b) {
        // median av hele bildet som fyllverdi
        std::vector<uint8_t> v;
        v.reserve(gray.total());
        for (int y = 0; y < h; ++y) {
            const uint8_t* p = gray.ptr<uint8_t>(y);
            v.insert(v.end(), p, p + w);
        }
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        const int fill = v[v.size() / 2];
        cv::Mat padded;
        cv::copyMakeBorder(crop, padded, pad_t, pad_b, pad_l, pad_r,
                           cv::BORDER_CONSTANT, fill);
        return padded;
    }
    return crop.clone();
}

cv::Mat annular_mask(int radius_px, double r_lo, double r_hi) {
    const int size = 2 * radius_px + 1;
    cv::Mat mask = cv::Mat::zeros(size, size, CV_8U);
    cv::circle(mask, {radius_px, radius_px}, static_cast<int>(r_hi), 255, cv::FILLED);
    cv::circle(mask, {radius_px, radius_px}, std::max(0, static_cast<int>(r_lo) - 1), 0,
               cv::FILLED);
    return mask;
}

// _crescent_mask: synlig maanesigd (inni kandidat, utenfor dominant).
cv::Mat crescent_mask(int radius_px, double r_lo, double r_hi,
                      double dom_dx, double dom_dy, double dom_r) {
    const int size = 2 * radius_px + 1;
    cv::Mat mask = cv::Mat::zeros(size, size, CV_8U);
    const int c = radius_px;
    for (int y = 0; y < size; ++y) {
        uint8_t* p = mask.ptr<uint8_t>(y);
        for (int x = 0; x < size; ++x) {
            const double r2 = (x - c) * (x - c) + (y - c) * (y - c);
            const double d2 = (x - c - dom_dx) * (x - c - dom_dx) +
                              (y - c - dom_dy) * (y - c - dom_dy);
            if (r2 <= r_hi * r_hi && d2 >= dom_r * dom_r && r2 > r_lo * r_lo)
                p[x] = 255;
        }
    }
    return mask;
}

// Mest isolert eksisterende treff (kilde for template).
const Hit* most_isolated(const std::vector<Hit>& hits) {
    const Hit* best = nullptr;
    double best_iso = -1;
    for (const auto& h : hits) {
        double dmin = 1e18;
        for (const auto& o : hits) {
            if (&o == &h) continue;
            dmin = std::min(dmin, std::hypot(h.x - o.x, h.y - o.y));
        }
        if (hits.size() == 1) dmin = 0;
        const double iso = dmin * h.score;
        if (iso > best_iso) { best_iso = iso; best = &h; }
    }
    return best;
}

} // namespace

std::vector<Hit> find_overlap_hits(const cv::Mat& gray, const std::vector<Hit>& hits,
                                   const Calibration& calib, const Config& cfg) {
    if (static_cast<int>(hits.size()) >= cfg.hit_overlap_trigger || hits.empty()) return {};

    const double cx0 = calib.center.x, cy0 = calib.center.y;
    const double delta = calib.delta_px, R10 = calib.R10_px;
    const double search_r = cfg.hit_search_r_max_frac * (R10 + 9.0 * delta);
    const double marker_r = cfg.hit_marker_radius_frac * delta;
    const double dot_r = cfg.hit_dot_radius_frac * delta;

    // 1) Stemmekart
    VoteMap vm = circle_vote_map(gray, calib.center, search_r, marker_r, dot_r, cfg, 0.0);
    double amax;
    cv::minMaxLoc(vm.acc, nullptr, &amax);
    if (amax < 1e-6) return {};

    // 2) Lokale maksima med LITEN NMS
    const int small_nms = odd(cfg.hit_overlap_nms_frac * marker_r);
    cv::Mat dil;
    cv::dilate(vm.acc, dil,
               cv::getStructuringElement(cv::MORPH_ELLIPSE, {small_nms, small_nms}));
    const double vote_floor = cfg.hit_overlap_vote_frac * amax;
    std::vector<cv::Point> peaks;
    for (int y = 0; y < vm.acc.rows; ++y) {
        const float* pa = vm.acc.ptr<float>(y);
        const float* pd = dil.ptr<float>(y);
        for (int x = 0; x < vm.acc.cols; ++x)
            if (pa[x] == pd[x] && pa[x] >= vote_floor) peaks.emplace_back(x, y);
    }

    // 3) NCC-template fra mest isolert treff
    const Hit* src = most_isolated(hits);
    if (!src) return {};
    const double r_lo = cfg.hit_overlap_tmpl_r_lo * marker_r;
    const double r_hi = cfg.hit_overlap_tmpl_r_hi * marker_r;
    const int tmpl_r = static_cast<int>(r_hi) + 2;
    cv::Mat tmpl_f;
    extract_patch(gray, src->x, src->y, tmpl_r).convertTo(tmpl_f, CV_32F, 1.0 / 255.0);

    // 4) Kandidat-loekke
    const double max_dist = cfg.hit_overlap_max_dist_frac * marker_r;
    const double min_offset = cfg.hit_overlap_min_offset_frac * marker_r;
    const double min_dist_other = cfg.hit_min_dist_frac * marker_r;
    const double max_anchor_r = cfg.hit_overlap_max_anchor_r_frac * delta;

    std::vector<Hit> new_hits;
    std::vector<const Hit*> already;
    for (const auto& h : hits) already.push_back(&h);

    for (const auto& anchor : hits) {
        if (std::hypot(anchor.x - cx0, anchor.y - cy0) > max_anchor_r) continue;
        for (const auto& pk : peaks) {
            const double X = pk.x + vm.offset.x, Y = pk.y + vm.offset.y;
            const double d_anchor = std::hypot(X - anchor.x, Y - anchor.y);
            if (d_anchor < min_offset || d_anchor > max_dist) continue;
            bool too_close = false;
            for (const Hit* k : already) {
                if (k == &anchor) continue;
                if (std::hypot(X - k->x, Y - k->y) < min_dist_other) { too_close = true; break; }
            }
            for (const auto& nh : new_hits)
                if (std::hypot(X - nh.x, Y - nh.y) < min_dist_other) { too_close = true; break; }
            if (too_close) continue;
            if (std::hypot(X - cx0, Y - cy0) > search_r) continue;

            const cv::Mat cmask = crescent_mask(tmpl_r, r_lo, r_hi,
                                                anchor.x - X, anchor.y - Y, marker_r);
            if (cv::countNonZero(cmask) < 10) continue;
            cv::Mat cand_f;
            extract_patch(gray, X, Y, tmpl_r).convertTo(cand_f, CV_32F, 1.0 / 255.0);
            if (cand_f.size() != tmpl_f.size()) continue;
            cv::Mat cmask_f;
            cmask.convertTo(cmask_f, CV_32F, 1.0 / 255.0);
            cv::Mat res;
            cv::matchTemplate(cand_f, tmpl_f, res, cv::TM_CCOEFF_NORMED, cmask_f);
            double ncc;
            cv::minMaxLoc(res, nullptr, &ncc);
            if (!std::isfinite(ncc) || ncc < cfg.hit_overlap_ncc_thresh) continue;

            new_hits.push_back({X, Y, anchor.type, ncc});
        }
    }

    // Dedupliser nye treff mot hverandre (hoeyeste score foerst)
    std::sort(new_hits.begin(), new_hits.end(),
              [](const Hit& a, const Hit& b) { return a.score > b.score; });
    std::vector<Hit> deduped;
    for (const auto& h : new_hits) {
        bool ok = true;
        for (const auto& k : deduped)
            if (std::hypot(h.x - k.x, h.y - k.y) < min_dist_other) { ok = false; break; }
        if (ok) deduped.push_back(h);
    }
    return deduped;
}

std::vector<Hit> find_center_hits(const cv::Mat& gray, const std::vector<Hit>& hits,
                                  const Calibration& calib, const Config& cfg) {
    if (static_cast<int>(hits.size()) >= cfg.hit_overlap_trigger || hits.empty()) return {};

    const double cx0 = calib.center.x, cy0 = calib.center.y;
    const double delta = calib.delta_px, R10 = calib.R10_px;
    const double marker_r = cfg.hit_marker_radius_frac * delta;
    const double scan_r = cfg.hit_center_scan_r_frac * R10;
    const double min_dist = cfg.hit_min_dist_frac * marker_r;

    const Hit* src = most_isolated(hits);
    if (!src) return {};
    const double r_lo = cfg.hit_overlap_tmpl_r_lo * marker_r;
    const double r_hi = cfg.hit_overlap_tmpl_r_hi * marker_r;
    const int tmpl_r = static_cast<int>(r_hi) + 2;
    const int tmpl_size = 2 * tmpl_r + 1;

    cv::Mat tmask = annular_mask(tmpl_r, r_lo, r_hi);
    cv::Mat tmask_f;
    tmask.convertTo(tmask_f, CV_32F, 1.0 / 255.0);
    cv::Mat tmpl_f;
    extract_patch(gray, src->x, src->y, tmpl_r).convertTo(tmpl_f, CV_32F, 1.0 / 255.0);

    const int h_img = gray.rows, w_img = gray.cols;
    const int x0 = std::max(0, static_cast<int>(cx0 - scan_r) - tmpl_r);
    const int y0 = std::max(0, static_cast<int>(cy0 - scan_r) - tmpl_r);
    const int x1 = std::min(w_img, static_cast<int>(cx0 + scan_r) + tmpl_r + 1);
    const int y1 = std::min(h_img, static_cast<int>(cy0 + scan_r) + tmpl_r + 1);
    if (x1 - x0 < tmpl_size || y1 - y0 < tmpl_size) return {};
    cv::Mat roi_f;
    gray(cv::Rect(x0, y0, x1 - x0, y1 - y0)).convertTo(roi_f, CV_32F, 1.0 / 255.0);

    cv::Mat corr;
    cv::matchTemplate(roi_f, tmpl_f, corr, cv::TM_CCOEFF_NORMED, tmask_f);

    const int nms_k = odd(min_dist);
    cv::Mat dil;
    cv::dilate(corr, dil, cv::getStructuringElement(cv::MORPH_ELLIPSE, {nms_k, nms_k}));

    std::vector<Hit> candidates;
    for (int ry = 0; ry < corr.rows; ++ry) {
        const float* pc = corr.ptr<float>(ry);
        const float* pd = dil.ptr<float>(ry);
        for (int rx = 0; rx < corr.cols; ++rx) {
            if (pc[rx] != pd[rx] || pc[rx] < cfg.hit_center_ncc_thresh ||
                !std::isfinite(pc[rx])) continue;
            const double X = x0 + rx + tmpl_r, Y = y0 + ry + tmpl_r;
            if (std::hypot(X - cx0, Y - cy0) > scan_r) continue;
            bool near_existing = false;
            for (const auto& h : hits)
                if (std::hypot(X - h.x, Y - h.y) < min_dist) { near_existing = true; break; }
            if (near_existing) continue;
            candidates.push_back({X, Y, 'f', pc[rx]});
        }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Hit& a, const Hit& b) { return a.score > b.score; });
    std::vector<Hit> kept;
    for (const auto& c : candidates) {
        bool ok = true;
        for (const auto& k : kept)
            if (std::hypot(c.x - k.x, c.y - k.y) < min_dist) { ok = false; break; }
        if (ok) kept.push_back(c);
    }
    return kept;
}

} // namespace bestefar
