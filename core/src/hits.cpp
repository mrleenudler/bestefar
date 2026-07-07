#include "hits.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <optional>

#include "circles.h"
#include "numpy_compat.h"
#include "overlap.h"

// Midlertidig feilsoking: BF_DEBUG_HITS=1
static bool hits_debug() {
    static const bool on = std::getenv("BF_DEBUG_HITS") != nullptr;
    return on;
}

namespace bestefar {
namespace {

// _annulus_percentile: intensitets-persentil q i annulus [r_in, r_out].
std::optional<double> annulus_percentile(const cv::Mat& gray, double cx, double cy,
                                         double r_in, double r_out, double q) {
    const int h = gray.rows, w = gray.cols;
    const int x0 = std::max(0, static_cast<int>(cx - r_out));
    const int x1 = std::min(w, static_cast<int>(cx + r_out) + 1);
    const int y0 = std::max(0, static_cast<int>(cy - r_out));
    const int y1 = std::min(h, static_cast<int>(cy + r_out) + 1);
    if (x1 <= x0 || y1 <= y0) return std::nullopt;
    std::vector<float> vals;
    const double ri2 = r_in * r_in, ro2 = r_out * r_out;
    for (int y = y0; y < y1; ++y) {
        const uint8_t* p = gray.ptr<uint8_t>(y);
        for (int x = x0; x < x1; ++x) {
            const double rr2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            if (rr2 >= ri2 && rr2 <= ro2) vals.push_back(p[x]);
        }
    }
    if (vals.empty()) return std::nullopt;
    return npc::percentile(std::move(vals), q);
}

// _refine_on_dot: glatt bort rutenettet, finn ekstremum, sentroid i naboskapet.
cv::Point2d refine_on_dot(const cv::Mat& gray, double cx, double cy, double dot_r,
                          char marker_type) {
    const int h = gray.rows, w = gray.cols;
    const int win = static_cast<int>(std::lround(dot_r * 2.0));
    const int x0 = std::max(0, static_cast<int>(std::lround(cx)) - win);
    const int x1 = std::min(w, static_cast<int>(std::lround(cx)) + win + 1);
    const int y0 = std::max(0, static_cast<int>(std::lround(cy)) - win);
    const int y1 = std::min(h, static_cast<int>(std::lround(cy)) + win + 1);
    if (x1 <= x0 || y1 <= y0) return {cx, cy};
    cv::Mat patch;
    gray(cv::Rect(x0, y0, x1 - x0, y1 - y0)).convertTo(patch, CV_32F);
    cv::Mat sm;
    cv::GaussianBlur(patch, sm, cv::Size(0, 0), std::max(1.0, dot_r * 0.5));
    cv::Mat dev;
    double mn, mx;
    cv::minMaxLoc(sm, &mn, &mx);
    if (marker_type == 'f') dev = mx - sm;    // moerk prikk -> hoeyt avvik
    else dev = sm - mn;                        // lys prikk
    cv::Point pk;
    cv::minMaxLoc(dev, nullptr, nullptr, nullptr, &pk);

    double sw = 0, sx = 0, sy = 0;
    const double dr2 = dot_r * dot_r;
    for (int y = 0; y < dev.rows; ++y) {
        const float* p = dev.ptr<float>(y);
        for (int x = 0; x < dev.cols; ++x) {
            if ((x - pk.x) * (x - pk.x) + (y - pk.y) * (y - pk.y) > dr2) continue;
            const double wgt = static_cast<double>(p[x]) * p[x];
            sw += wgt; sx += x * wgt; sy += y * wgt;
        }
    }
    if (sw < 1e-6) return {cx, cy};
    return {x0 + sx / sw, y0 + sy / sw};
}

// _refine_white: velg polaritet etter prikken, sentrer paa den.
std::tuple<double, double, char> refine_white(const cv::Mat& gray, double x, double y,
                                              double marker_r, double dot_r) {
    auto disc = annulus_percentile(gray, x, y, marker_r * 0.35, marker_r * 0.75, 50);
    auto dot = annulus_percentile(gray, x, y, 0.0, dot_r * 0.7, 50);
    const char mtype = (disc && dot && *dot <= *disc) ? 'f' : 'o';
    cv::Point2d r = refine_on_dot(gray, x, y, dot_r, mtype);
    if (std::hypot(r.x - x, r.y - y) > dot_r * 1.5) r = {x, y};
    return {r.x, r.y, mtype};
}

} // namespace

std::vector<Hit> detect_hits(const cv::Mat& gray, const Calibration& calib, const Config& cfg) {
    const double delta = calib.delta_px;
    const double R10 = calib.R10_px;
    const double r_ring1 = R10 + 9.0 * delta;
    const double search_r = cfg.hit_search_r_max_frac * r_ring1;
    const double marker_r = cfg.hit_marker_radius_frac * delta;
    const double dot_r = cfg.hit_dot_radius_frac * delta;

    // Enhetlig stemme-detektor over HELE skiva
    const double inner_r = cfg.hit_unified_inner_frac * delta;
    auto circ = detect_circles(gray, calib.center, search_r, marker_r, dot_r, cfg, inner_r);

    std::vector<Hit> hits;
    for (const auto& c : circ) {
        auto [rx, ry, mtype] = refine_white(gray, c.x, c.y, marker_r, dot_r);
        hits.push_back({rx, ry, mtype, c.score});
    }

    // Dedup: behold hoeyest score innen min avstand
    const double min_dist = cfg.hit_min_dist_frac * marker_r;
    std::sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) {
        return a.score > b.score;
    });
    std::vector<Hit> kept;
    for (const auto& t : hits) {
        bool ok = true;
        for (const auto& k : kept)
            if (std::hypot(t.x - k.x, t.y - k.y) < min_dist) { ok = false; break; }
        if (ok) kept.push_back(t);
    }

    if (hits_debug()) {
        std::fprintf(stderr, "HITS enhetlig+dedup: %zu\n", kept.size());
        for (const auto& t : kept)
            std::fprintf(stderr, "  (%.1f,%.1f) type=%c score=%.2f\n",
                         t.x, t.y, t.type, t.score);
    }

    // Overlapp-pass + sentrum-sveip (overlap.py)
    if (cfg.hit_overlap_pass) {
        auto extra = find_overlap_hits(gray, kept, calib, cfg);
        if (hits_debug())
            for (const auto& t : extra)
                std::fprintf(stderr, "HITS overlapp: (%.1f,%.1f) ncc=%.2f\n",
                             t.x, t.y, t.score);
        kept.insert(kept.end(), extra.begin(), extra.end());
    }
    if (cfg.hit_center_scan) {
        auto extra = find_center_hits(gray, kept, calib, cfg);
        if (hits_debug())
            for (const auto& t : extra)
                std::fprintf(stderr, "HITS sentrum: (%.1f,%.1f) ncc=%.2f\n",
                             t.x, t.y, t.score);
        kept.insert(kept.end(), extra.begin(), extra.end());
    }
    return kept;
}

} // namespace bestefar
