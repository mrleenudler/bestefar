#include "screen.h"

#include <array>
#include <cmath>
#include <vector>

#include "numpy_compat.h"

namespace bestefar {
namespace {

using Quad = std::array<cv::Point2f, 4>;

// screen.py _order_corners: (TL, TR, BR, BL)
Quad order_corners(std::array<cv::Point2f, 4> p) {
    std::sort(p.begin(), p.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        return a.y < b.y;
    });
    cv::Point2f tl = p[0], tr = p[1], bl = p[2], br = p[3];
    if (tl.x > tr.x) std::swap(tl, tr);
    if (bl.x > br.x) std::swap(bl, br);
    return {tl, tr, br, bl};
}

// multiotsu3: 3-klasse multi-Otsu paa histogram (screen.py). Brukes ikke i
// den aktive stien (kun hysterese-otsu), utelatt.

// Linje {p : n·p = c}
struct Line { cv::Point2d n; double c; };

std::optional<cv::Point2d> intersect(const Line& l1, const Line& l2) {
    const double det = l1.n.x * l2.n.y - l1.n.y * l2.n.x;
    if (std::abs(det) < 1e-6) return std::nullopt;
    return cv::Point2d((l1.c * l2.n.y - l2.c * l1.n.y) / det,
                       (l1.n.x * l2.c - l2.n.x * l1.c) / det);
}

// _fit_line_robust: vektet TLS med IRLS (Huber 2 runder -> Tukey).
struct RobustFit { Line line; std::vector<double> resid; };
std::optional<RobustFit> fit_line_robust(const std::vector<cv::Point2d>& P,
                                         const std::vector<double>& w0, int n_iter = 6) {
    if (P.size() < 8) return std::nullopt;
    std::vector<double> w = w0;
    Line ln{};
    std::vector<double> resid(P.size(), 0.0);
    for (int it = 0; it < n_iter; ++it) {
        double wsum = 0, mx = 0, my = 0;
        for (size_t i = 0; i < P.size(); ++i) { wsum += w[i]; mx += w[i] * P[i].x; my += w[i] * P[i].y; }
        if (wsum < 1e-9) return std::nullopt;
        mx /= wsum; my /= wsum;
        double sxx = 0, sxy = 0, syy = 0;
        for (size_t i = 0; i < P.size(); ++i) {
            const double qx = P[i].x - mx, qy = P[i].y - my;
            sxx += w[i] * qx * qx; sxy += w[i] * qx * qy; syy += w[i] * qy * qy;
        }
        sxx /= wsum; sxy /= wsum; syy /= wsum;
        // Minste egenvektor til 2x2 symmetrisk matrise
        const double tr = sxx + syy, det = sxx * syy - sxy * sxy;
        const double disc = std::sqrt(std::max(0.0, tr * tr / 4.0 - det));
        const double l_min = tr / 2.0 - disc;
        cv::Point2d n;
        if (std::abs(sxy) > 1e-12) {
            n = cv::Point2d(l_min - syy, sxy);
        } else {
            n = (sxx <= syy) ? cv::Point2d(1, 0) : cv::Point2d(0, 1);
        }
        const double nn = std::hypot(n.x, n.y);
        n.x /= nn; n.y /= nn;
        const double c = n.x * mx + n.y * my;
        ln = {n, c};

        std::vector<double> ar(P.size());
        for (size_t i = 0; i < P.size(); ++i) {
            resid[i] = std::abs(P[i].x * n.x + P[i].y * n.y - c);
            ar[i] = resid[i];
        }
        std::nth_element(ar.begin(), ar.begin() + ar.size() / 2, ar.end());
        const double med = ar[ar.size() / 2];
        const double sigma = std::max(1.4826 * med, 0.5);
        for (size_t i = 0; i < P.size(); ++i) {
            double rw;
            if (it < 2) {
                rw = std::min(1.0, (1.345 * sigma) / std::max(resid[i], 1e-6));   // Huber
            } else {
                const double u = resid[i] / (4.685 * sigma);                       // Tukey
                rw = u < 1.0 ? (1.0 - u * u) * (1.0 - u * u) : 0.0;
            }
            w[i] = w0[i] * rw;
        }
    }
    return RobustFit{ln, resid};
}

// _snap_line_to_edge: la siden vandre mot YTTERSTE STERKE gradient-kam.
std::optional<Line> snap_line_to_edge(Line line, cv::Point2d A, cv::Point2d B,
                                      const cv::Point2d& center,
                                      const cv::Mat& gx, const cv::Mat& gy,
                                      const cv::Mat& gmag, const cv::Mat& roi,
                                      const Config& cfg) {
    const int H = gmag.rows, W = gmag.cols;
    const int mx_side = std::max(H, W);
    const int band = std::max(4, static_cast<int>(cfg.screen_snap_band_frac * mx_side));
    const double dir_cos = cfg.screen_snap_dir_cos;
    const double skip = cfg.screen_snap_end_skip_frac;
    const double strong_frac = cfg.screen_snap_strong_frac;

    for (int iter = 0; iter < cfg.screen_snap_iters; ++iter) {
        const cv::Point2d A_p = A - (A.dot(line.n) - line.c) * line.n;
        const cv::Point2d B_p = B - (B.dot(line.n) - line.c) * line.n;
        const cv::Point2d seg = B_p - A_p;
        const double Lp = std::hypot(seg.x, seg.y);
        if (Lp < 8) return std::nullopt;
        const cv::Point2d t = seg / Lp;
        cv::Point2d nrm(-t.y, t.x);
        int m0 = static_cast<int>(skip * Lp), m1 = static_cast<int>((1.0 - skip) * Lp);
        if (m1 - m0 < 8) { m0 = 0; m1 = static_cast<int>(Lp); }

        // "Utover" = retningen som oeker avstand fra skjermsenteret
        const cv::Point2d mid = A_p + 0.5 * (m0 + m1) * t;
        const double out_sign = nrm.dot(mid - center) >= 0 ? 1.0 : -1.0;

        std::vector<cv::Point2d> edge_pts;
        std::vector<double> edge_w;
        for (int m = m0; m < m1; ++m) {
            const cv::Point2d base = A_p + static_cast<double>(m) * t;
            double best_signed = -1e18, best_g = -1.0;
            cv::Point2d best_pt;
            // Foerst: finn radens maks gyldige gradient
            double row_max = -1.0;
            for (int d = -band; d <= band; ++d) {
                const cv::Point2d q = base + static_cast<double>(d) * nrm;
                const int xi = std::clamp(static_cast<int>(std::lround(q.x)), 0, W - 1);
                const int yi = std::clamp(static_cast<int>(std::lround(q.y)), 0, H - 1);
                if (!roi.empty() && roi.at<uint8_t>(yi, xi) == 0) continue;
                const float gv = gmag.at<float>(yi, xi);
                const float gxv = gx.at<float>(yi, xi), gyv = gy.at<float>(yi, xi);
                const double align = std::abs((gxv * nrm.x + gyv * nrm.y) /
                                              (std::hypot(gxv, gyv) + 1e-6));
                if (align < dir_cos) continue;
                row_max = std::max(row_max, static_cast<double>(gv));
            }
            if (row_max <= 0) continue;
            // Saa: ytterste punkt >= strong_frac * radmaks
            for (int d = -band; d <= band; ++d) {
                const cv::Point2d q = base + static_cast<double>(d) * nrm;
                const int xi = std::clamp(static_cast<int>(std::lround(q.x)), 0, W - 1);
                const int yi = std::clamp(static_cast<int>(std::lround(q.y)), 0, H - 1);
                if (!roi.empty() && roi.at<uint8_t>(yi, xi) == 0) continue;
                const float gv = gmag.at<float>(yi, xi);
                if (gv < strong_frac * row_max) continue;
                const float gxv = gx.at<float>(yi, xi), gyv = gy.at<float>(yi, xi);
                const double align = std::abs((gxv * nrm.x + gyv * nrm.y) /
                                              (std::hypot(gxv, gyv) + 1e-6));
                if (align < dir_cos) continue;
                const double signed_d = d * out_sign;
                if (signed_d > best_signed) { best_signed = signed_d; best_g = gv; best_pt = q; }
            }
            if (best_g > 0) { edge_pts.push_back(best_pt); edge_w.push_back(best_g); }
        }
        if (edge_pts.size() < 10) return std::nullopt;
        auto fit = fit_line_robust(edge_pts, edge_w);
        if (!fit) return std::nullopt;
        line = fit->line;
    }
    return line;
}

// _screen_blob: hysterese-terskling i ROI, morph-close, stoerste komponent.
std::optional<std::vector<cv::Point2f>> screen_blob(const cv::Mat& gray_blur,
                                                    const Config& cfg, const cv::Mat& roi,
                                                    double* area_frac_out) {
    double otsu;
    if (!roi.empty()) {
        std::vector<uint8_t> vals;
        for (int y = 0; y < gray_blur.rows; ++y)
            for (int x = 0; x < gray_blur.cols; ++x)
                if (roi.at<uint8_t>(y, x) > 0) vals.push_back(gray_blur.at<uint8_t>(y, x));
        if (vals.size() < 50) return std::nullopt;
        cv::Mat vm(static_cast<int>(vals.size()), 1, CV_8U, vals.data());
        cv::Mat dummy;
        otsu = cv::threshold(vm, dummy, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    } else {
        cv::Mat dummy;
        otsu = cv::threshold(gray_blur, dummy, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    }

    const double t_low = cfg.screen_low_frac * otsu;
    cv::Mat ml = gray_blur >= t_low;      // lav terskel
    cv::Mat mh = gray_blur >= otsu;       // froe
    if (!roi.empty()) {
        ml.setTo(0, roi == 0);
        cv::bitwise_and(mh, roi > 0, mh);
    }

    cv::Mat labels;
    const int num = cv::connectedComponents(ml, labels, 8, CV_32S);
    if (num <= 1) return std::nullopt;
    // Stoerste komponent som inneholder et froe
    std::vector<int64_t> sizes(num, 0);
    std::vector<bool> seeded(num, false);
    for (int y = 0; y < labels.rows; ++y) {
        const int32_t* pl = labels.ptr<int32_t>(y);
        const uint8_t* ph = mh.ptr<uint8_t>(y);
        for (int x = 0; x < labels.cols; ++x) {
            ++sizes[pl[x]];
            if (ph[x]) seeded[pl[x]] = true;
        }
    }
    int best = -1;
    for (int l = 1; l < num; ++l)
        if (seeded[l] && (best < 0 || sizes[l] > sizes[best])) best = l;
    if (best < 0) return std::nullopt;

    cv::Mat mask = (labels == best);
    const int k = std::max(3, static_cast<int>(0.04 * std::max(gray_blur.rows, gray_blur.cols)));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {k, k}));
    // (screen_blob_open_frac og convex hull er av i produksjon — ikke portert)

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    if (contours.empty()) return std::nullopt;
    size_t ci = 0;
    double best_area = -1;
    for (size_t i = 0; i < contours.size(); ++i) {
        const double a = cv::contourArea(contours[i]);
        if (a > best_area) { best_area = a; ci = i; }
    }
    const double area_frac = best_area / (static_cast<double>(gray_blur.rows) * gray_blur.cols);
    if (area_frac_out) *area_frac_out = area_frac;
    if (area_frac < cfg.screen_min_area_frac) return std::nullopt;

    std::vector<cv::Point2f> out;
    out.reserve(contours[ci].size());
    for (const auto& p : contours[ci]) out.emplace_back(static_cast<float>(p.x),
                                                        static_cast<float>(p.y));
    return out;
}

// _rough_quad: approxPolyDP med oekende epsilon, ellers ekstrempunkter.
Quad rough_quad(const std::vector<cv::Point2f>& contour) {
    const double peri = cv::arcLength(contour, true);
    for (int i = 0; i < 8; ++i) {
        const double eps_frac = 0.01 + i * 0.01;
        std::vector<cv::Point2f> approx;
        cv::approxPolyDP(contour, approx, eps_frac * peri, true);
        if (approx.size() == 4 && cv::isContourConvex(approx))
            return order_corners({approx[0], approx[1], approx[2], approx[3]});
    }
    size_t i_smin = 0, i_smax = 0, i_dmin = 0, i_dmax = 0;
    for (size_t i = 0; i < contour.size(); ++i) {
        const float s = contour[i].x + contour[i].y;
        const float d = contour[i].x - contour[i].y;
        if (s < contour[i_smin].x + contour[i_smin].y) i_smin = i;
        if (s > contour[i_smax].x + contour[i_smax].y) i_smax = i;
        if (d < contour[i_dmin].x - contour[i_dmin].y) i_dmin = i;
        if (d > contour[i_dmax].x - contour[i_dmax].y) i_dmax = i;
    }
    return order_corners({contour[i_smin], contour[i_dmax], contour[i_smax], contour[i_dmin]});
}

double seg_dist(const cv::Point2f& p, const cv::Point2f& A, const cv::Point2f& B) {
    const cv::Point2f AB = B - A;
    const double L2 = AB.dot(AB);
    if (L2 < 1e-9) return cv::norm(p - A);
    const double t = std::clamp(static_cast<double>((p - A).dot(AB)) / L2, 0.0, 1.0);
    const cv::Point2f proj = A + static_cast<float>(t) * AB;
    return cv::norm(p - proj);
}

// _refine_from_contour + kant-snapping.
std::optional<Quad> refine_from_contour(const std::vector<cv::Point2f>& contour,
                                        const Quad& rough, const cv::Mat& gmag,
                                        const cv::Mat& gx, const cv::Mat& gy,
                                        const cv::Mat& roi, const Config& cfg) {
    const cv::Point2f tl = rough[0], tr = rough[1], br = rough[2], bl = rough[3];
    const std::array<std::pair<cv::Point2f, cv::Point2f>, 4> seg{{
        {tl, tr}, {tr, br}, {br, bl}, {bl, tl}}};

    // Tilordne konturpunkter til naermeste side
    std::vector<int> assign(contour.size());
    for (size_t i = 0; i < contour.size(); ++i) {
        double dmin = 1e18;
        for (int s = 0; s < 4; ++s) {
            const double d = seg_dist(contour[i], seg[s].first, seg[s].second);
            if (d < dmin) { dmin = d; assign[i] = s; }
        }
    }

    const int H = gmag.rows, W = gmag.cols;
    std::array<Line, 4> lines;
    for (int s = 0; s < 4; ++s) {
        std::vector<cv::Point2d> P;
        std::vector<double> w;
        for (size_t i = 0; i < contour.size(); ++i) {
            if (assign[i] != s) continue;
            P.emplace_back(contour[i].x, contour[i].y);
            const int xi = std::clamp(static_cast<int>(std::lround(contour[i].x)), 0, W - 1);
            const int yi = std::clamp(static_cast<int>(std::lround(contour[i].y)), 0, H - 1);
            w.push_back(gmag.at<float>(yi, xi) + 1e-3);
        }
        const double L = cv::norm(seg[s].second - seg[s].first);
        if (P.size() < 10 || L < 8) return std::nullopt;
        // (screen_side_outer_tol_frac er av i produksjon — ikke portert)

        auto fit = fit_line_robust(P, w);
        if (!fit) return std::nullopt;

        // Spenn-primaer godkjenning (inlier-andel + spenn)
        const double tol = std::max(2.0, cfg.screen_side_inlier_tol_frac * L);
        const cv::Point2f t2 = (seg[s].second - seg[s].first) / static_cast<float>(L);
        int n_inl = 0;
        double tp_min = 1e18, tp_max = -1e18;
        for (size_t i = 0; i < P.size(); ++i) {
            if (fit->resid[i] > tol) continue;
            ++n_inl;
            const double tp = P[i].x * t2.x + P[i].y * t2.y;
            tp_min = std::min(tp_min, tp);
            tp_max = std::max(tp_max, tp);
        }
        const double frac = static_cast<double>(n_inl) / P.size();
        const double span = n_inl >= 2 ? (tp_max - tp_min) / L : 0.0;
        if (frac < cfg.screen_side_min_inlier_frac || span < cfg.screen_side_min_span_frac)
            return std::nullopt;
        lines[s] = fit->line;
    }

    auto TL = intersect(lines[3], lines[0]);
    auto TR = intersect(lines[0], lines[1]);
    auto BR = intersect(lines[1], lines[2]);
    auto BL = intersect(lines[2], lines[3]);
    if (!TL || !TR || !BR || !BL) return std::nullopt;
    const Quad orig_quad = order_corners({cv::Point2f(*TL), cv::Point2f(*TR),
                                          cv::Point2f(*BR), cv::Point2f(*BL)});

    // Kant-snapping (C10-fiksen): hver side vandrer mot ytterste sterke kam.
    if (cfg.screen_refine_gradient_lines && !gx.empty()) {
        cv::Point2d q_center(0, 0);
        for (const auto& p : orig_quad) { q_center.x += p.x / 4.0; q_center.y += p.y / 4.0; }
        const std::array<std::tuple<Line, cv::Point2d, cv::Point2d>, 4> sides{{
            {lines[0], cv::Point2d(*TL), cv::Point2d(*TR)},
            {lines[1], cv::Point2d(*TR), cv::Point2d(*BR)},
            {lines[2], cv::Point2d(*BR), cv::Point2d(*BL)},
            {lines[3], cv::Point2d(*BL), cv::Point2d(*TL)}}};
        std::array<Line, 4> snapped;
        for (int s = 0; s < 4; ++s) {
            auto sl = snap_line_to_edge(std::get<0>(sides[s]), std::get<1>(sides[s]),
                                        std::get<2>(sides[s]), q_center, gx, gy, gmag, roi, cfg);
            snapped[s] = sl ? *sl : std::get<0>(sides[s]);
        }
        auto sTL = intersect(snapped[3], snapped[0]);
        auto sTR = intersect(snapped[0], snapped[1]);
        auto sBR = intersect(snapped[1], snapped[2]);
        auto sBL = intersect(snapped[2], snapped[3]);
        if (sTL && sTR && sBR && sBL) {
            const Quad snap_quad = order_corners({cv::Point2f(*sTL), cv::Point2f(*sTR),
                                                  cv::Point2f(*sBR), cv::Point2f(*sBL)});
            const double diag = cv::norm(orig_quad[2] - orig_quad[0]) + 1e-6;
            double shift = 0;
            for (int i = 0; i < 4; ++i)
                shift = std::max(shift, static_cast<double>(cv::norm(snap_quad[i] - orig_quad[i])));
            if (shift < 0.25 * diag) return snap_quad;
        }
    }
    return orig_quad;
}

} // namespace

std::optional<cv::Rect> screen_blob_box(const cv::Mat& gray_blur, const Config& cfg,
                                        const cv::Mat& roi) {
    auto contour = screen_blob(gray_blur, cfg, roi, nullptr);
    if (!contour) return std::nullopt;
    return cv::boundingRect(*contour);
}

cv::Mat normalize_stretch(const cv::Mat& gray, const Config& cfg) {
    std::vector<float> v;
    v.reserve(gray.total());
    for (int y = 0; y < gray.rows; ++y) {
        const uint8_t* p = gray.ptr<uint8_t>(y);
        for (int x = 0; x < gray.cols; ++x) v.push_back(p[x]);
    }
    const double pct = cfg.screen_stretch_pct;
    std::vector<float> v2 = v;
    const double lo = npc::percentile(std::move(v), pct);
    const double hi = npc::percentile(std::move(v2), 100.0 - pct);
    if (hi - lo < 1.0) return gray.clone();
    cv::Mat out;
    gray.convertTo(out, CV_32F, 1.0, -lo);
    out *= 255.0 / (hi - lo);
    cv::Mat u8;
    out.convertTo(u8, CV_8U);  // convertTo klipper til [0,255] (saturate_cast)
    return u8;
}

cv::Mat apparatus_roi(const cv::Mat& gray, const Config& cfg) {
    const int H = gray.rows, W = gray.cols;
    const int mx = std::max(H, W);
    int win = std::max(5, static_cast<int>(cfg.screen_contrast_win_frac * mx));
    if (win % 2 == 0) ++win;
    cv::Mat g;
    gray.convertTo(g, CV_32F);
    cv::Mat mean, msq;
    cv::boxFilter(g, mean, -1, {win, win});
    cv::boxFilter(g.mul(g), msq, -1, {win, win});
    cv::Mat var = msq - mean.mul(mean);
    cv::max(var, 0.0, var);
    cv::Mat std_;
    cv::sqrt(var, std_);
    cv::Mat stdn;
    cv::normalize(std_, stdn, 0, 255, cv::NORM_MINMAX);
    stdn.convertTo(stdn, CV_8U);
    cv::Mat hi;
    cv::threshold(stdn, hi, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    const int kc = std::max(3, static_cast<int>(cfg.screen_contrast_win_frac * mx));
    cv::morphologyEx(hi, hi, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {kc, kc}));
    const int ko = std::max(3, static_cast<int>(cfg.screen_roi_open_frac * mx));
    cv::morphologyEx(hi, hi, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {ko, ko}));

    cv::Mat labels, stats, centroids;
    const int num = cv::connectedComponentsWithStats(hi, labels, stats, centroids, 8);
    if (num <= 1) return {};

    // Stoerste strukturerte region med en lys (skjerm-)piksel
    std::vector<int> order;
    for (int l = 1; l < num; ++l) order.push_back(l);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return stats.at<int32_t>(a, cv::CC_STAT_AREA) > stats.at<int32_t>(b, cv::CC_STAT_AREA);
    });
    int seed = -1;
    for (int l : order) {
        uint8_t vmax = 0;
        for (int y = 0; y < H; ++y) {
            const int32_t* pl = labels.ptr<int32_t>(y);
            const uint8_t* pg = gray.ptr<uint8_t>(y);
            for (int x = 0; x < W; ++x)
                if (pl[x] == l) vmax = std::max(vmax, pg[x]);
        }
        if (vmax >= cfg.screen_roi_bright_min) { seed = l; break; }
    }
    if (seed < 0) return {};

    cv::Mat mask = (labels == seed);
    // Fyll regionens ytre kontur (ikke konveks hylle — den "vandrer")
    std::vector<std::vector<cv::Point>> cnts;
    cv::findContours(mask, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat roi = cv::Mat::zeros(H, W, CV_8U);
    cv::drawContours(roi, cnts, -1, 1, cv::FILLED);
    const int d = std::max(3, static_cast<int>(cfg.screen_roi_dilate_frac * mx));
    cv::dilate(roi, roi, cv::getStructuringElement(cv::MORPH_ELLIPSE, {d, d}));
    return roi;
}

std::optional<ScreenRect> rectify_to_screen(const cv::Mat& img_bgr, const Config& cfg) {
    const int h0 = img_bgr.rows, w0 = img_bgr.cols;
    const double scale = std::min(static_cast<double>(cfg.screen_work_size) /
                                  std::max(h0, w0), 1.0);
    cv::Mat small;
    cv::resize(img_bgr, small, cv::Size(static_cast<int>(w0 * scale),
                                        static_cast<int>(h0 * scale)),
               0, 0, cv::INTER_AREA);
    cv::Mat gray;
    cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
    gray = normalize_stretch(gray, cfg);
    cv::Mat grayb;
    cv::GaussianBlur(gray, grayb, cv::Size(0, 0), cfg.screen_blur_sigma);

    cv::Mat roi;
    if (cfg.screen_use_contrast_roi) {
        roi = apparatus_roi(gray, cfg);
        if (roi.empty()) return std::nullopt;
    }

    double area_frac = 0;
    auto contour = screen_blob(grayb, cfg, roi, &area_frac);
    if (!contour) return std::nullopt;

    const Quad rough = rough_quad(*contour);

    cv::Mat gx, gy, gmag;
    cv::Sobel(grayb, gx, CV_32F, 1, 0, 3);
    cv::Sobel(grayb, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, gmag);

    auto refined = refine_from_contour(*contour, rough, gmag, gx, gy, roi, cfg);
    if (!refined) return std::nullopt;

    Quad rect;
    for (int i = 0; i < 4; ++i)
        rect[i] = cv::Point2f(static_cast<float>((*refined)[i].x / scale),
                              static_cast<float>((*refined)[i].y / scale));

    const auto [tl, tr, br, bl] = rect;
    const double wA = cv::norm(br - bl), wB = cv::norm(tr - tl);
    const double hA = cv::norm(tr - br), hB = cv::norm(tl - bl);
    const int Wd = static_cast<int>(std::lround(std::max(wA, wB)));
    const int Hd = static_cast<int>(std::lround(std::max(hA, hB)));
    if (Wd < 50 || Hd < 50) return std::nullopt;

    const std::array<cv::Point2f, 4> dst{{
        {0, 0}, {static_cast<float>(Wd - 1), 0},
        {static_cast<float>(Wd - 1), static_cast<float>(Hd - 1)},
        {0, static_cast<float>(Hd - 1)}}};
    const cv::Mat M = cv::getPerspectiveTransform(rect.data(), dst.data());
    ScreenRect out;
    cv::warpPerspective(img_bgr, out.warped, M, {Wd, Hd}, cv::INTER_LINEAR);
    out.M = M;
    out.quad = rect;
    return out;
}

} // namespace bestefar
