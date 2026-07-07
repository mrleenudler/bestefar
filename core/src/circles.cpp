#include "circles.h"

#include <cmath>

#include "numpy_compat.h"

namespace bestefar {
namespace {

int odd(double v) {
    int k = static_cast<int>(std::lround(v));
    if (k % 2 == 0) k += 1;
    return std::max(3, k);
}

} // namespace

VoteMap circle_vote_map(const cv::Mat& gray, cv::Point2d center, double search_r,
                        double marker_r, double dot_r, const Config& cfg,
                        double inner_r) {
    const double cx = center.x, cy = center.y;
    const int h = gray.rows, w = gray.cols;
    const int x0 = std::max(0, static_cast<int>(cx - search_r));
    const int x1 = std::min(w, static_cast<int>(cx + search_r) + 1);
    const int y0 = std::max(0, static_cast<int>(cy - search_r));
    const int y1 = std::min(h, static_cast<int>(cy + search_r) + 1);
    cv::Mat roi = gray(cv::Rect(x0, y0, x1 - x0, y1 - y0));

    // 1) LED-rutenett-avstoying: valgfri gauss-forblur + N x median
    const int kmed = odd(cfg.circ_median_frac * marker_r);
    cv::Mat med;
    const double pg = cfg.circ_pre_gauss_frac * marker_r;
    if (pg > 0) cv::GaussianBlur(roi, med, cv::Size(0, 0), pg);
    else med = roi.clone();
    for (int i = 0; i < cfg.circ_median_passes; ++i)
        cv::medianBlur(med, med, std::max(3, kmed));

    // 2) gradient
    cv::Mat gx, gy, mag;
    cv::Sobel(med, gx, CV_32F, 1, 0, 3);
    cv::Sobel(med, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, mag);

    // lav terskel; kompletthet siler
    std::vector<float> magv;
    magv.reserve(mag.total());
    for (int r = 0; r < mag.rows; ++r) {
        const float* p = mag.ptr<float>(r);
        for (int c = 0; c < mag.cols; ++c) magv.push_back(p[c]);
    }
    const double thr = npc::percentile(std::move(magv), cfg.circ_edge_pct);

    // kanter + retningsgating mot maalsenter
    struct Edge { float x, y, ux, uy; };
    std::vector<Edge> edges;
    const double reject = cfg.circ_radial_reject;
    const double exempt_r = cfg.circ_center_exempt_frac * marker_r;
    for (int r = 0; r < mag.rows; ++r) {
        const float* pm = mag.ptr<float>(r);
        const float* px = gx.ptr<float>(r);
        const float* py = gy.ptr<float>(r);
        for (int c = 0; c < mag.cols; ++c) {
            if (pm[c] <= thr) continue;
            const float ux = px[c] / pm[c], uy = py[c] / pm[c];
            const double dcx = cx - (c + x0), dcy = cy - (r + y0);
            const double dn = std::hypot(dcx, dcy) + 1e-6;
            if (reject < 1.0) {
                const double align = std::abs(ux * (dcx / dn) + uy * (dcy / dn));
                const bool near_c = dn < exempt_r;
                if (!near_c && align >= reject) continue;
            }
            if (inner_r > 0 && dn < inner_r) continue;
            edges.push_back({static_cast<float>(c), static_cast<float>(r), ux, uy});
        }
    }

    cv::Mat acc = cv::Mat::zeros(roi.size(), CV_32F);
    const int Hh = roi.rows, Ww = roi.cols;
    const int nr = cfg.circ_n_radii;
    std::vector<std::pair<double, double>> rw;   // (radius, vekt)
    for (int i = 0; i < nr; ++i)
        rw.emplace_back((0.82 + (1.18 - 0.82) * i / std::max(nr - 1, 1)) * marker_r, 1.0);
    if (dot_r > 2.0) {
        const int nd = std::max(2, nr / 2);
        for (int i = 0; i < nd; ++i)
            rw.emplace_back((0.80 + (1.20 - 0.80) * i / std::max(nd - 1, 1)) * dot_r,
                            cfg.circ_dot_vote_weight);
    }
    for (const auto& [r, wgt] : rw) {
        for (const double sgn : {+1.0, -1.0}) {
            for (const auto& e : edges) {
                const int vx = static_cast<int>(std::lround(e.x + sgn * r * e.ux));
                const int vy = static_cast<int>(std::lround(e.y + sgn * r * e.uy));
                if (vx < 0 || vx >= Ww || vy < 0 || vy >= Hh) continue;
                acc.at<float>(vy, vx) += static_cast<float>(wgt);
            }
        }
    }

    cv::GaussianBlur(acc, acc, cv::Size(0, 0),
                     std::max(1.0, cfg.circ_acc_blur_frac * marker_r));
    return {acc, {x0, y0}};
}

std::vector<CircleCand> detect_circles(const cv::Mat& gray, cv::Point2d center,
                                       double search_r, double marker_r, double dot_r,
                                       const Config& cfg, double inner_r) {
    VoteMap vm = circle_vote_map(gray, center, search_r, marker_r, dot_r, cfg, inner_r);
    cv::Mat& acc = vm.acc;
    const int x0 = vm.offset.x, y0 = vm.offset.y;

    const int nms_r = odd(cfg.circ_nms_frac * marker_r);
    cv::Mat dil;
    cv::dilate(acc, dil, cv::getStructuringElement(cv::MORPH_ELLIPSE, {nms_r, nms_r}));

    double amax;
    cv::minMaxLoc(acc, nullptr, &amax);
    const double floor_ = cfg.circ_min_votes_frac * marker_r;
    const double thr_keep = std::max(cfg.circ_peak_min_frac * amax, floor_);

    std::vector<CircleCand> cands;
    if (amax > 0) {
        for (int py = 0; py < acc.rows; ++py) {
            const float* pa = acc.ptr<float>(py);
            const float* pd = dil.ptr<float>(py);
            for (int px = 0; px < acc.cols; ++px) {
                if (pa[px] != pd[px] || pa[px] <= 0 || pa[px] < thr_keep) continue;
                const double X = px + x0, Y = py + y0;
                const double d = std::hypot(X - center.x, Y - center.y);
                if (d > search_r || d < inner_r) continue;
                cands.push_back({X, Y, pa[px] / amax});
            }
        }
    }
    std::sort(cands.begin(), cands.end(),
              [](const CircleCand& a, const CircleCand& b) { return a.score > b.score; });
    return cands;
}

} // namespace bestefar
