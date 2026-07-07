#include "outer_circle.h"

#include <cstdio>
#include <cstdlib>
#include <random>

#include "numpy_compat.h"
#include "preprocess.h"

// Midlertidig feilsoking: BF_DEBUG_OUTER=1
static bool outer_debug() {
    static const bool on = std::getenv("BF_DEBUG_OUTER") != nullptr;
    return on;
}

namespace bestefar {
namespace {

struct EdgePoints {
    std::vector<float> x, y, ux, uy, mag;
    size_t size() const { return x.size(); }
};

// points.extract_edge_points
std::optional<EdgePoints> extract_edge_points(const cv::Mat& mag, const cv::Mat& ux,
                                              const cv::Mat& uy, const Config& cfg) {
    std::vector<float> nonzero;
    nonzero.reserve(static_cast<size_t>(mag.total()) / 4);
    for (int r = 0; r < mag.rows; ++r) {
        const float* p = mag.ptr<float>(r);
        for (int c = 0; c < mag.cols; ++c)
            if (p[c] > cfg.outer_circle_mag_floor) nonzero.push_back(p[c]);
    }
    if (nonzero.empty()) return std::nullopt;
    const size_t n_nonzero = nonzero.size();
    const double threshold = npc::percentile(std::move(nonzero), cfg.outer_circle_mag_percentile);

    const double margin = cfg.outer_circle_border_margin_frac * std::min(mag.cols, mag.rows);
    EdgePoints pts;
    for (int r = 0; r < mag.rows; ++r) {
        const float* pm = mag.ptr<float>(r);
        const float* pux = ux.ptr<float>(r);
        const float* puy = uy.ptr<float>(r);
        for (int c = 0; c < mag.cols; ++c) {
            if (pm[c] < threshold) continue;
            if (c < margin || c >= mag.cols - margin || r < margin || r >= mag.rows - margin)
                continue;
            pts.x.push_back(static_cast<float>(c));
            pts.y.push_back(static_cast<float>(r));
            pts.ux.push_back(pux[c]);
            pts.uy.push_back(puy[c]);
            pts.mag.push_back(pm[c]);
        }
    }

    if (outer_debug()) {
        double msum = 0;
        for (float m : pts.mag) msum += m;
        std::fprintf(stderr, "EXTRACT: n_nonzero=%zu thr=%.3f n_foer_subsample=%zu "
                     "mag_middel=%.2f\n", n_nonzero, threshold, pts.size(),
                     pts.size() ? msum / pts.size() : 0.0);
    }

    // Random subsample (numpy rng.choice uten replacement -> partiell Fisher-Yates).
    // max_edge_points <= 0 betyr INGEN subsampling (se merknad i config.h).
    if (cfg.outer_circle_max_edge_points <= 0) return pts;
    const size_t max_points = static_cast<size_t>(cfg.outer_circle_max_edge_points);
    if (pts.size() > max_points) {
        std::mt19937_64 rng(42);
        std::vector<size_t> idx(pts.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
        for (size_t i = 0; i < max_points; ++i) {
            std::uniform_int_distribution<size_t> d(i, idx.size() - 1);
            std::swap(idx[i], idx[d(rng)]);
        }
        EdgePoints sub;
        sub.x.reserve(max_points); sub.y.reserve(max_points);
        sub.ux.reserve(max_points); sub.uy.reserve(max_points); sub.mag.reserve(max_points);
        for (size_t i = 0; i < max_points; ++i) {
            const size_t j = idx[i];
            sub.x.push_back(pts.x[j]); sub.y.push_back(pts.y[j]);
            sub.ux.push_back(pts.ux[j]); sub.uy.push_back(pts.uy[j]);
            sub.mag.push_back(pts.mag[j]);
        }
        return sub;
    }
    return pts;
}

// voting.intersection_vote (uten akkumulator-retur; peak + subpixel-centroid)
cv::Point2d intersection_vote(const EdgePoints& pts, double cross_min,
                              const Config& cfg, int H, int W) {
    cv::Mat acc = cv::Mat::zeros(H, W, CV_32F);
    const int n = static_cast<int>(pts.size());
    if (n < 2) return {W / 2.0, H / 2.0};

    unsigned seed = 42;
    if (const char* s = std::getenv("BF_VOTE_SEED")) seed = std::atoi(s);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> pick(0, n - 1);
    int M = cfg.outer_circle_center_pairs;
    if (const char* s = std::getenv("BF_VOTE_PAIRS")) M = std::atoi(s);
    const double parallel_eps = cfg.outer_circle_parallel_eps;
    const double max_dist_frac = cfg.outer_circle_max_center_distance_frac;
    const double max_dist = max_dist_frac > 0 ? max_dist_frac * std::hypot(W, H) : -1.0;
    const double icx = W / 2.0, icy = H / 2.0;

    for (int m = 0; m < M; ++m) {
        int i = pick(rng), j = pick(rng);
        if (i == j) j = (j + 1) % n;
        const double ux1 = pts.ux[i], uy1 = pts.uy[i];
        const double ux2 = pts.ux[j], uy2 = pts.uy[j];
        const double cross = ux1 * uy2 - uy1 * ux2;
        if (std::abs(cross) < parallel_eps || std::abs(cross) < cross_min) continue;
        const double dx = pts.x[j] - pts.x[i], dy = pts.y[j] - pts.y[i];
        const double s = (dx * uy2 - dy * ux2) / cross;
        const double cx = pts.x[i] + s * ux1;
        const double cy = pts.y[i] + s * uy1;
        if (max_dist > 0 && std::hypot(cx - icx, cy - icy) > max_dist) continue;
        const int xi = static_cast<int>(std::lround(cx));
        const int yi = static_cast<int>(std::lround(cy));
        if (xi < 0 || xi >= W || yi < 0 || yi >= H) continue;
        acc.at<float>(yi, xi) += std::min(pts.mag[i], pts.mag[j]);
    }

    // Peak + vektet centroid i vindu (voting.py subpixel refinement)
    cv::Point maxLoc;
    cv::minMaxLoc(acc, nullptr, nullptr, nullptr, &maxLoc);
    const int win = cfg.outer_circle_center_win;
    const int x0 = std::max(0, maxLoc.x - win), x1 = std::min(W, maxLoc.x + win + 1);
    const int y0 = std::max(0, maxLoc.y - win), y1 = std::min(H, maxLoc.y + win + 1);
    double sw = 0, sx = 0, sy = 0;
    for (int y = y0; y < y1; ++y) {
        const float* p = acc.ptr<float>(y);
        for (int x = x0; x < x1; ++x) { sw += p[x]; sx += x * p[x]; sy += y * p[x]; }
    }
    if (sw > 0) return {sx / sw, sy / sw};
    return {static_cast<double>(maxLoc.x), static_cast<double>(maxLoc.y)};
}

// histogram.build_radius_histogram + smooth. Returnerer (hist_s, bin_width).
std::vector<double> radius_histogram(const EdgePoints& pts, const cv::Point2d& c,
                                     const Config& cfg) {
    float maxx = 0, maxy = 0;
    for (size_t i = 0; i < pts.size(); ++i) {
        maxx = std::max(maxx, pts.x[i]);
        maxy = std::max(maxy, pts.y[i]);
    }
    const double rmax_search = 0.6 * std::min(static_cast<int>(maxx) + 1,
                                              static_cast<int>(maxy) + 1);
    const double bw = cfg.outer_circle_r_bin_px;
    const int nbins = std::max(1, static_cast<int>(std::floor(rmax_search / bw)));
    std::vector<double> hist(nbins, 0.0);
    for (size_t i = 0; i < pts.size(); ++i) {
        const double dx = pts.x[i] - c.x, dy = pts.y[i] - c.y;
        const double ri = std::sqrt(dx * dx + dy * dy);
        const double rs = ri + 1e-6;
        const double ai = std::abs((dx / rs) * pts.ux[i] + (dy / rs) * pts.uy[i]);
        if (ai < cfg.outer_circle_align_min) continue;
        const int b = static_cast<int>(ri / bw);
        if (b >= 0 && b < nbins) hist[b] += pts.mag[i] * ai;
    }
    return npc::gaussian_filter1d(hist, cfg.outer_circle_r_smooth_sigma);
}

struct PeakBand {
    double r_peak, r_lo, r_hi;
};

// histogram.select_accepted_peaks (find_peaks -> cluster -> FWHM -> coverage)
std::vector<PeakBand> select_accepted_peaks(const EdgePoints& pts, const cv::Point2d& c,
                                            const std::vector<double>& hist_s,
                                            const Config& cfg) {
    const double bw = cfg.outer_circle_r_bin_px;
    double hmax = 0;
    for (double v : hist_s) hmax = std::max(hmax, v);
    std::vector<int> peaks = npc::local_maxima(hist_s, cfg.outer_circle_peak_score_min_frac * hmax);
    if (peaks.empty()) {
        int am = 0;
        for (int i = 1; i < static_cast<int>(hist_s.size()); ++i)
            if (hist_s[i] > hist_s[am]) am = i;
        peaks = {am};
    }

    // cluster_peaks: sorter paa radius synkende, slaa sammen naerliggende
    struct PR { int idx; double r; };
    std::vector<PR> prs;
    for (int p : peaks) prs.push_back({p, (p + 0.5) * bw});
    std::sort(prs.begin(), prs.end(), [](const PR& a, const PR& b) { return a.r > b.r; });
    const double cluster_px = cfg.outer_circle_peaks_cluster_px;
    std::vector<bool> used(prs.size(), false);
    std::vector<PR> clustered;
    for (size_t i = 0; i < prs.size(); ++i) {
        if (used[i]) continue;
        std::vector<size_t> cluster{i};
        for (size_t j = i + 1; j < prs.size(); ++j)
            if (!used[j] && std::abs(prs[i].r - prs[j].r) < cluster_px) {
                cluster.push_back(j);
                used[j] = true;
            }
        if (cluster.size() == 1) {
            clustered.push_back(prs[i]);
        } else {
            double tw = 0, rw = 0, best_v = -1;
            int best_idx = prs[i].idx;
            for (size_t k : cluster) {
                const double v = hist_s[prs[k].idx];
                tw += v; rw += prs[k].r * v;
                if (v > best_v) { best_v = v; best_idx = prs[k].idx; }
            }
            if (tw > 0) clustered.push_back({best_idx, rw / tw});
            else clustered.push_back(prs[i]);
        }
    }

    // Radii for alle punkter (for coverage)
    std::vector<double> ri(pts.size());
    for (size_t i = 0; i < pts.size(); ++i)
        ri[i] = std::hypot(pts.x[i] - c.x, pts.y[i] - c.y);

    std::vector<PeakBand> accepted;
    for (const auto& pr : clustered) {
        // peak_fwhm_band
        const double half = 0.5 * hist_s[pr.idx];
        int left = pr.idx;
        while (left > 0 && hist_s[left] >= half) --left;
        ++left;
        int right = pr.idx;
        while (right < static_cast<int>(hist_s.size()) - 1 && hist_s[right] >= half) ++right;
        --right;
        const double r_lo = left * bw;
        const double r_hi = (right + 1) * bw;
        if (right - left + 1 < cfg.outer_circle_fwhm_min_bins) continue;

        // angular coverage i FWHM-baandet (120 bins hardkodet, som histogram.py)
        int n_in = 0;
        std::vector<bool> bin_hit(120, false);
        for (size_t i = 0; i < pts.size(); ++i) {
            if (ri[i] < r_lo || ri[i] > r_hi) continue;
            ++n_in;
            const double th = std::atan2(pts.y[i] - c.y, pts.x[i] - c.x);
            int b = static_cast<int>(std::floor((th + CV_PI) / (2 * CV_PI) * 120));
            b = std::clamp(b, 0, 119);
            bin_hit[b] = true;
        }
        if (n_in < 10) continue;
        int uniq = 0;
        for (bool bh : bin_hit) uniq += bh ? 1 : 0;
        const double coverage = static_cast<double>(uniq) / 120.0;
        if (coverage >= cfg.outer_circle_cov_min_frac)
            accepted.push_back({pr.r, r_lo, r_hi});
    }
    return accepted;
}

} // namespace

cv::Point2d vote_test(const VoteTestPoints& vpts, const Config& cfg, int H, int W) {
    EdgePoints pts;
    pts.x = vpts.x; pts.y = vpts.y; pts.ux = vpts.ux; pts.uy = vpts.uy; pts.mag = vpts.mag;
    return intersection_vote(pts, cfg.outer_circle_cross_min_pass1, cfg, H, W);
}

std::optional<OuterCircle> detect_outer_circle(const cv::Mat& img_bgr, const Config& cfg) {
    // 1) downscale -> gray -> blur -> gradienter
    auto [img_down, scale] = preprocess::downscale_max_side(img_bgr, cfg.outer_circle_max_side);
    cv::Mat gray = preprocess::to_gray(img_down);
    cv::Mat blur = preprocess::gaussian_blur(gray, cfg.outer_circle_blur_sigma);
    auto g = preprocess::compute_gradients(blur);
    const int H = gray.rows, W = gray.cols;

    // 2) Pass 1: grovsenter med akse-filtrering
    cv::Mat mag_pass1 = preprocess::suppress_axis_normals(g.ux, g.uy, g.mag,
                                                          cfg.outer_circle_filter_angle_deg);
    auto points1 = extract_edge_points(mag_pass1, g.ux, g.uy, cfg);
    if (!points1) return std::nullopt;
    if (outer_debug()) {
        if (FILE* f = std::fopen("_cpp_points1.txt", "w")) {
            for (size_t i = 0; i < points1->size(); ++i)
                std::fprintf(f, "%.6f %.6f %.6f %.6f %.6f\n", points1->x[i], points1->y[i],
                             points1->ux[i], points1->uy[i], points1->mag[i]);
            std::fclose(f);
        }
    }
    const cv::Point2d c0 = intersection_vote(*points1, cfg.outer_circle_cross_min_pass1, cfg, H, W);

    auto hist1_s = radius_histogram(*points1, c0, cfg);
    auto accepted = select_accepted_peaks(*points1, c0, hist1_s, cfg);
    if (outer_debug()) {
        std::fprintf(stderr, "OUTER: n1=%zu c0=(%.1f,%.1f) aksepterte=%zu:",
                     points1->size(), c0.x, c0.y, accepted.size());
        for (const auto& b : accepted)
            std::fprintf(stderr, " [r=%.0f lo=%.0f hi=%.0f]", b.r_peak, b.r_lo, b.r_hi);
        std::fprintf(stderr, "\n");
        if (FILE* f = std::fopen("_cpp_hist1.txt", "w")) {
            for (double v : hist1_s) std::fprintf(f, "%.6f\n", v);
            std::fclose(f);
        }
    }

    // 3) Pass 2: presist senter (radius-baand + alignment-filter)
    auto points2_raw = extract_edge_points(g.mag, g.ux, g.uy, cfg);
    if (!points2_raw) return std::nullopt;

    if (accepted.empty()) {
        int am = 0;
        for (int i = 1; i < static_cast<int>(hist1_s.size()); ++i)
            if (hist1_s[i] > hist1_s[am]) am = i;
        const double r_peak = (am + 0.5) * cfg.outer_circle_r_bin_px;
        accepted = {{r_peak, r_peak - 5, r_peak + 5}};
    }

    double r_outer = 0;
    for (const auto& b : accepted) r_outer = std::max(r_outer, b.r_peak);
    const double outer_cut = r_outer + cfg.outer_circle_pass2_outer_cut_eps;

    EdgePoints points2;
    for (size_t i = 0; i < points2_raw->size(); ++i) {
        const double dx = points2_raw->x[i] - c0.x, dy = points2_raw->y[i] - c0.y;
        const double ri = std::sqrt(dx * dx + dy * dy);
        bool in_band = false;
        for (const auto& b : accepted)
            if (ri >= b.r_lo && ri <= b.r_hi) { in_band = true; break; }
        if (!in_band || ri > outer_cut) continue;
        const double rs = ri + 1e-6;
        const double ai = std::abs((dx / rs) * points2_raw->ux[i] + (dy / rs) * points2_raw->uy[i]);
        if (ai < cfg.outer_circle_align_min) continue;
        points2.x.push_back(points2_raw->x[i]);
        points2.y.push_back(points2_raw->y[i]);
        points2.ux.push_back(points2_raw->ux[i]);
        points2.uy.push_back(points2_raw->uy[i]);
        points2.mag.push_back(points2_raw->mag[i]);
    }
    const cv::Point2d c1 = intersection_vote(points2, cfg.outer_circle_cross_min_pass2, cfg, H, W);

    // 5) Endelig radius fra pass2-histogram om c1 (refine er av i produksjon)
    auto hist2_s = radius_histogram(points2, c1, cfg);
    int am = 0;
    for (int i = 1; i < static_cast<int>(hist2_s.size()); ++i)
        if (hist2_s[i] > hist2_s[am]) am = i;
    const double r_final = (am + 0.5) * cfg.outer_circle_r_bin_px;

    return OuterCircle{c1.x / scale, c1.y / scale, r_final / scale};
}

} // namespace bestefar
