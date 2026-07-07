#include "rings.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "numpy_compat.h"

// Midlertidig feilsokings-sporing: sett BF_DEBUG_RINGS=1 i miljoet.
static bool rings_debug() {
    static const bool on = std::getenv("BF_DEBUG_RINGS") != nullptr;
    return on;
}

namespace bestefar {
namespace {

// warp_polar_gray: P[theta_rad, rho_col]
cv::Mat warp_polar_gray(const cv::Mat& gray, cv::Point2d center, double r_max,
                        int r_samples, int theta_samples) {
    cv::Mat gf;
    gray.convertTo(gf, CV_32F);
    cv::Mat P;
    cv::warpPolar(gf, P, cv::Size(r_samples, theta_samples),
                  cv::Point2f(static_cast<float>(center.x), static_cast<float>(center.y)),
                  r_max, cv::WARP_POLAR_LINEAR | cv::INTER_LINEAR);
    return P;
}

cv::Mat radial_gradient_abs(const cv::Mat& P, double sigma_rho) {
    cv::Mat Pb = P;
    if (sigma_rho > 0)
        cv::GaussianBlur(P, Pb, cv::Size(0, 0), sigma_rho, 1.0);
    cv::Mat g;
    cv::Sobel(Pb, g, CV_32F, 1, 0, 3);
    return cv::abs(g);
}

// estimate_spacing_autocorr med halv-lag (oktav)-sjekk og subpixel.
std::optional<double> estimate_spacing_autocorr(const std::vector<double>& H,
                                                int min_lag, int max_lag) {
    const int n = static_cast<int>(H.size());
    double mean = 0;
    for (double v : H) mean += v;
    mean /= n;
    std::vector<double> Hc(n);
    for (int i = 0; i < n; ++i) Hc[i] = H[i] - mean;
    // autokorrelasjon ac[lag] = sum Hc[i]*Hc[i+lag]
    std::vector<double> ac(n, 0.0);
    for (int lag = 0; lag < n; ++lag) {
        double s = 0;
        for (int i = 0; i + lag < n; ++i) s += Hc[i] * Hc[i + lag];
        ac[lag] = s;
    }
    max_lag = std::min(max_lag, n - 2);
    if (max_lag <= min_lag) return std::nullopt;
    int lag = min_lag;
    for (int i = min_lag; i < max_lag; ++i)
        if (ac[i] > ac[lag]) lag = i;

    while (lag / 2 >= min_lag) {
        const int half = lag / 2;
        const int lo = std::max(min_lag, half - 3), hi = std::min(max_lag, half + 4);
        int local = lo;
        for (int i = lo; i < hi; ++i)
            if (ac[i] > ac[local]) local = i;
        if (ac[local] > 0.5 * ac[lag] && std::abs(local - half) <= 3) lag = local;
        else break;
    }
    double out = lag;
    if (lag >= 1 && lag < n - 1)
        out += npc::subpixel_parabola(ac[lag - 1], ac[lag], ac[lag + 1]);
    return out;
}

// find_profile_peaks: >= venstre, > hoyre; graadig etter hoyde m/min-avstand.
std::vector<std::pair<double, double>> find_profile_peaks(const std::vector<double>& H,
                                                          double min_frac, int min_sep) {
    const int n = static_cast<int>(H.size());
    double hmax = 0;
    for (double v : H) hmax = std::max(hmax, v);
    const double thresh = min_frac * hmax;
    std::vector<int> idx;
    for (int i = 1; i + 1 < n; ++i)
        if (H[i] >= thresh && H[i] >= H[i - 1] && H[i] > H[i + 1]) idx.push_back(i);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return H[a] > H[b]; });
    std::vector<int> kept;
    for (int i : idx) {
        bool ok = true;
        for (int j : kept)
            if (std::abs(i - j) < min_sep) { ok = false; break; }
        if (ok) kept.push_back(i);
    }
    std::vector<std::pair<double, double>> peaks;
    for (int i : kept)
        peaks.emplace_back(i + npc::subpixel_parabola(H[i - 1], H[i], H[i + 1]), H[i]);
    std::sort(peaks.begin(), peaks.end());
    return peaks;
}

struct ProgressionFit {
    double a = 0, delta = 0;
    std::vector<int> k;
    std::vector<bool> inlier;
    std::vector<double> rho;
};

// fit_ring_progression: iterativ LSQ med fjerning av verste residual.
std::optional<ProgressionFit> fit_ring_progression(std::vector<double> peak_rho,
                                                   double delta0, const Config& cfg) {
    std::sort(peak_rho.begin(), peak_rho.end(), std::greater<double>());
    const int n = static_cast<int>(peak_rho.size());
    if (n < 3 || delta0 <= 0) return std::nullopt;

    std::vector<int> k(n);
    for (int i = 0; i < n; ++i)
        k[i] = static_cast<int>(std::lround((peak_rho[0] - peak_rho[i]) / delta0));

    std::vector<bool> sel(n, true);
    double a = peak_rho[0], d = delta0;
    const double max_resid = std::max(cfg.ring_fit_max_resid_frac * delta0, 2.0);
    for (int round = 0; round < n; ++round) {
        // LSQ rho = a - d*k over inliers
        int m = 0;
        double Sk = 0, S1 = 0, Skk = 0, Sr = 0, Skr = 0;
        for (int i = 0; i < n; ++i) {
            if (!sel[i]) continue;
            ++m;
            S1 += 1; Sk += -k[i]; Skk += static_cast<double>(k[i]) * k[i];
            Sr += peak_rho[i]; Skr += -static_cast<double>(k[i]) * peak_rho[i];
        }
        if (m < 3) break;
        const double det = S1 * Skk - Sk * Sk;
        if (std::abs(det) < 1e-12) break;
        a = (Sr * Skk - Sk * Skr) / det;
        d = (S1 * Skr - Sk * Sr) / det;
        // verste residual
        int worst = -1;
        double worst_r = -1;
        for (int i = 0; i < n; ++i) {
            if (!sel[i]) continue;
            const double r = std::abs(peak_rho[i] - (a - d * k[i]));
            if (r > worst_r) { worst_r = r; worst = i; }
        }
        if (worst_r <= max_resid) break;
        sel[worst] = false;
    }
    int m = 0;
    for (bool s : sel) m += s ? 1 : 0;
    if (d <= 0 || m < 3) return std::nullopt;
    ProgressionFit fit;
    fit.a = a; fit.delta = d; fit.k = k; fit.inlier = sel; fit.rho = peak_rho;
    return fit;
}

// _fit_harmonics: robust vektet LSQ av x(theta) = C+B1 sin+D1 cos+B2 sin2+D2 cos2.
struct HarmFit { double beta[5]; double rmse; std::vector<bool> mask; };
std::optional<HarmFit> fit_harmonics(const std::vector<double>& theta,
                                     const std::vector<double>& x,
                                     const std::vector<double>& w,
                                     int n_reject_rounds, double reject_sigma) {
    const int n = static_cast<int>(theta.size());
    std::vector<bool> mask(n);
    int m0 = 0;
    for (int i = 0; i < n; ++i) {
        mask[i] = std::isfinite(x[i]) && w[i] > 0;
        m0 += mask[i] ? 1 : 0;
    }
    if (m0 < 16) return std::nullopt;

    cv::Mat beta;
    auto solve_masked = [&]() -> bool {
        int m = 0;
        for (int i = 0; i < n; ++i) m += mask[i] ? 1 : 0;
        if (m < 16) return false;
        cv::Mat A(m, 5, CV_64F), b(m, 1, CV_64F);
        int r = 0;
        for (int i = 0; i < n; ++i) {
            if (!mask[i]) continue;
            const double sw = std::sqrt(w[i]);
            double* pa = A.ptr<double>(r);
            pa[0] = sw; pa[1] = sw * std::sin(theta[i]); pa[2] = sw * std::cos(theta[i]);
            pa[3] = sw * std::sin(2 * theta[i]); pa[4] = sw * std::cos(2 * theta[i]);
            b.at<double>(r, 0) = sw * x[i];
            ++r;
        }
        return cv::solve(A, b, beta, cv::DECOMP_SVD);
    };

    for (int round = 0; round <= n_reject_rounds; ++round) {
        if (!solve_masked()) return std::nullopt;
        // residualer + sigma over maskerte
        double ss = 0;
        int m = 0;
        std::vector<double> resid(n);
        for (int i = 0; i < n; ++i) {
            const double pred = beta.at<double>(0) + beta.at<double>(1) * std::sin(theta[i]) +
                                beta.at<double>(2) * std::cos(theta[i]) +
                                beta.at<double>(3) * std::sin(2 * theta[i]) +
                                beta.at<double>(4) * std::cos(2 * theta[i]);
            resid[i] = x[i] - pred;
            if (mask[i]) { ss += resid[i] * resid[i]; ++m; }
        }
        const double s = std::sqrt(ss / m);
        std::vector<bool> new_mask(n);
        int nm = 0;
        bool changed = false;
        for (int i = 0; i < n; ++i) {
            new_mask[i] = mask[i] && std::abs(resid[i]) <= std::max(reject_sigma * s, 0.5);
            if (new_mask[i] != mask[i]) changed = true;
            nm += new_mask[i] ? 1 : 0;
        }
        if (!changed || nm < 16) break;
        mask = new_mask;
    }
    // endelig rmse over maskerte
    double ss = 0;
    int m = 0;
    for (int i = 0; i < n; ++i) {
        if (!mask[i]) continue;
        const double pred = beta.at<double>(0) + beta.at<double>(1) * std::sin(theta[i]) +
                            beta.at<double>(2) * std::cos(theta[i]) +
                            beta.at<double>(3) * std::sin(2 * theta[i]) +
                            beta.at<double>(4) * std::cos(2 * theta[i]);
        ss += (x[i] - pred) * (x[i] - pred);
        ++m;
    }
    HarmFit out;
    for (int i = 0; i < 5; ++i) out.beta[i] = beta.at<double>(i);
    out.rmse = std::sqrt(ss / std::max(m, 1));
    out.mask = mask;
    return out;
}

// _ring_track: vektet sentroid av |g| i baand rundt rho_c, per vinkelrad.
bool ring_track(const cv::Mat& G, double rho_c, double band_px,
                std::vector<double>& x_out, std::vector<double>& w_out,
                double min_mag_frac = 0.2) {
    const int theta_rows = G.rows, rho_cols = G.cols;
    const int lo = std::max(0, static_cast<int>(std::floor(rho_c - band_px)));
    const int hi = std::min(rho_cols, static_cast<int>(std::ceil(rho_c + band_px)) + 1);
    if (hi - lo < 3) return false;
    x_out.assign(theta_rows, std::numeric_limits<double>::quiet_NaN());
    w_out.assign(theta_rows, 0.0);
    std::vector<double> s(theta_rows, 0.0), sc(theta_rows, 0.0);
    double smax = 0;
    for (int r = 0; r < theta_rows; ++r) {
        const float* p = G.ptr<float>(r);
        for (int c = lo; c < hi; ++c) { s[r] += p[c]; sc[r] += p[c] * c; }
        smax = std::max(smax, s[r]);
    }
    for (int r = 0; r < theta_rows; ++r) {
        if (s[r] > min_mag_frac * smax) {
            x_out[r] = sc[r] / s[r];
            w_out[r] = s[r];
        }
    }
    return true;
}

} // namespace

std::optional<Calibration> calibrate_and_refine(const cv::Mat& gray_in, cv::Point2d center0,
                                                double r_outer_est, const Config& cfg,
                                                int refine_iters_override) {
    const int theta_samples = cfg.ring_theta_samples;
    const double sigma_rho = cfg.ring_rho_sigma;
    const int n_iters = refine_iters_override > 0 ? refine_iters_override
                                                  : cfg.ring_refine_iters;

    cv::Mat gray = gray_in;
    if (cfg.ring_pre_blur_sigma > 0) {
        cv::GaussianBlur(gray_in, gray, cv::Size(0, 0), cfg.ring_pre_blur_sigma);
    }

    cv::Point2d center = center0;
    const double base_r_max = cfg.ring_r_max_frac * r_outer_est;
    double r_max = base_r_max;
    if (cfg.ring_generous_rmax) {
        const double r_edge = std::min({center.x, gray.cols - center.x,
                                        center.y, gray.rows - center.y});
        r_max = std::max(base_r_max, std::min(cfg.ring_generous_mult * r_outer_est,
                                              cfg.ring_generous_edge_frac * r_edge));
    }
    const int r_samples = static_cast<int>(std::lround(r_max));
    const double rho_scale = r_max / r_samples;

    ProgressionFit fit;
    std::vector<std::tuple<int, HarmFit>> harmonics;   // (k_rel, fit)
    bool have = false;

    for (int it = 0; it < n_iters; ++it) {
        cv::Mat P = warp_polar_gray(gray, center, r_max, r_samples, theta_samples);
        cv::Mat G = radial_gradient_abs(P, sigma_rho);
        // H = kolonnegjennomsnitt over vinkel
        std::vector<double> H(G.cols, 0.0);
        for (int r = 0; r < G.rows; ++r) {
            const float* p = G.ptr<float>(r);
            for (int c = 0; c < G.cols; ++c) H[c] += p[c];
        }
        for (auto& v : H) v /= G.rows;

        auto delta0 = estimate_spacing_autocorr(H, static_cast<int>(0.03 * r_samples),
                                                static_cast<int>(0.35 * r_samples));
        if (!delta0) {
            if (rings_debug()) std::fprintf(stderr, "RINGS it%d: autocorr FEIL\n", it);
            return std::nullopt;
        }
        auto peaks = find_profile_peaks(
            H, cfg.ring_peak_min_frac,
            std::max(3, static_cast<int>(cfg.ring_peak_min_sep_frac * *delta0)));
        // Ignorer topper helt inntil senteret (markoerklynge)
        std::vector<double> prho;
        for (const auto& [pos, h] : peaks)
            if (pos > 0.4 * *delta0) prho.push_back(pos);
        auto pf = fit_ring_progression(prho, *delta0, cfg);
        if (rings_debug())
            std::fprintf(stderr, "RINGS it%d: delta0=%.2f peaks=%zu brukbare=%zu prog=%s"
                         " (delta=%.2f a=%.1f)\n",
                         it, *delta0, peaks.size(), prho.size(), pf ? "OK" : "FEIL",
                         pf ? pf->delta : 0.0, pf ? pf->a : 0.0);
        if (!pf) return std::nullopt;

        // Per-ring harmonisk fit -> senterkorreksjon
        const double band_frac = it == 0 ? 0.25 : 0.15;
        const double band_px = band_frac * pf->delta;
        std::vector<double> theta(theta_samples);
        for (int i = 0; i < theta_samples; ++i)
            theta[i] = 2.0 * CV_PI * i / theta_samples;

        std::vector<cv::Point2d> corrections;
        std::vector<double> weights;
        std::vector<std::tuple<int, HarmFit>> harms;
        for (size_t i = 0; i < pf->rho.size(); ++i) {
            if (!pf->inlier[i]) continue;
            const double rho_fit = pf->a - pf->delta * pf->k[i];
            std::vector<double> x, w;
            if (!ring_track(G, rho_fit, band_px, x, w)) continue;
            auto hf = fit_harmonics(theta, x, w, cfg.ring_harm_reject_rounds,
                                    cfg.ring_harm_reject_sigma);
            if (!hf) continue;
            double coverage = 0;
            for (bool b : hf->mask) coverage += b ? 1 : 0;
            coverage /= hf->mask.size();
            if (coverage < cfg.ring_refine_min_coverage) continue;
            const double C = hf->beta[0], B1 = hf->beta[1], D1 = hf->beta[2];
            corrections.emplace_back(D1 * rho_scale, B1 * rho_scale);
            const double w_inner = 1.0 / std::max(C, 1.0);
            weights.push_back(w_inner * coverage / (hf->rmse + 0.3));
            hf->rmse = hf->rmse;   // beholdes
            HarmFit hcopy = *hf;
            harms.emplace_back(pf->k[i], hcopy);
        }
        if (rings_debug())
            std::fprintf(stderr, "RINGS it%d: korreksjoner=%zu\n", it, corrections.size());
        if (corrections.size() < 2) return std::nullopt;

        double sw = 0;
        cv::Point2d corr(0, 0);
        for (size_t i = 0; i < corrections.size(); ++i) {
            corr += weights[i] * corrections[i];
            sw += weights[i];
        }
        corr /= sw;
        center += corr;

        fit = *pf;
        harmonics = std::move(harms);
        have = true;
        if (std::abs(corr.x) < 0.3 && std::abs(corr.y) < 0.3) break;
    }
    if (!have) return std::nullopt;

    const double delta_px = fit.delta * rho_scale;
    (void)delta_px;

    // Ringverdier: ytterste inlier = ring_value_outermost foer forankring
    int k_out = INT_MAX;
    for (const auto& [k_rel, hf] : harmonics) k_out = std::min(k_out, k_rel);

    std::sort(harmonics.begin(), harmonics.end(),
              [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });

    Calibration calib;
    calib.center = center;
    calib.r_max = r_max;
    calib.rho_scale = rho_scale;
    for (const auto& [k_rel, hf] : harmonics) {
        RingHarmonic rh;
        rh.value = cfg.ring_value_outermost + (k_rel - k_out);
        for (int i = 0; i < 5; ++i) rh.beta[i] = hf.beta[i] * rho_scale;
        rh.rmse = hf.rmse;
        double cov = 0;
        for (bool b : hf.mask) cov += b ? 1 : 0;
        rh.coverage = cov / hf.mask.size();
        calib.harmonics.push_back(rh);
        calib.ring_values.push_back(rh.value);
        calib.ring_radii_px.push_back(rh.beta[0]);
    }

    // Global LSQ: R(v) = R10 + (10-v)*delta
    {
        const int n = static_cast<int>(calib.ring_values.size());
        double S1 = n, Sk = 0, Skk = 0, Sr = 0, Skr = 0;
        for (int i = 0; i < n; ++i) {
            const double k = 10.0 - calib.ring_values[i];
            Sk += k; Skk += k * k;
            Sr += calib.ring_radii_px[i]; Skr += k * calib.ring_radii_px[i];
        }
        const double det = S1 * Skk - Sk * Sk;
        if (std::abs(det) < 1e-12) return std::nullopt;
        calib.R10_px = (Sr * Skk - Sk * Skr) / det;
        calib.delta_px = (S1 * Skr - Sk * Sr) / det;
    }

    // Verdi-forankring: R10 ~= ratio * delta
    if (cfg.ring_anchor_R10_eq_delta && calib.delta_px > 1e-6) {
        const int off = static_cast<int>(std::lround(cfg.ring_R10_over_delta -
                                                     calib.R10_px / calib.delta_px));
        if (off != 0) {
            for (auto& v : calib.ring_values) v += off;
            for (auto& h : calib.harmonics) h.value += off;
            calib.R10_px += off * calib.delta_px;
        }
    }
    return calib;
}

std::pair<bool, std::string> validate_calibration(const Calibration& calib, const Config& cfg) {
    std::string reasons;
    auto add = [&](const std::string& r) {
        if (!reasons.empty()) reasons += "; ";
        reasons += r;
    };
    if (calib.delta_px < cfg.gate_min_delta_px)
        add("ringavstand for liten");
    if (static_cast<int>(calib.harmonics.size()) < cfg.gate_min_rings)
        add("for faa ringer");
    if (calib.delta_px > 0 && calib.ring_radii_px.size() >= 2) {
        double max_resid = 0;
        for (size_t i = 0; i < calib.ring_values.size(); ++i) {
            const double pred = calib.R10_px + (10.0 - calib.ring_values[i]) * calib.delta_px;
            max_resid = std::max(max_resid, std::abs(calib.ring_radii_px[i] - pred));
        }
        if (max_resid / calib.delta_px > cfg.gate_max_resid_frac)
            add("ringene er ikke jevnt fordelt");
    }
    if (calib.R10_px <= 0)
        add("R10 ikke fysisk mulig");
    return {reasons.empty(), reasons};
}

std::pair<double, double> local_ring_geometry(const Calibration& calib, double theta) {
    const double basis[5] = {1.0, std::sin(theta), std::cos(theta),
                             std::sin(2 * theta), std::cos(2 * theta)};
    const int n = static_cast<int>(calib.harmonics.size());
    if (n < 2) return {calib.R10_px, calib.delta_px};
    double S1 = n, Sk = 0, Skk = 0, Sr = 0, Skr = 0;
    for (const auto& h : calib.harmonics) {
        double r = 0;
        for (int i = 0; i < 5; ++i) r += h.beta[i] * basis[i];
        const double k = 10.0 - h.value;
        Sk += k; Skk += k * k; Sr += r; Skr += k * r;
    }
    const double det = S1 * Skk - Sk * Sk;
    if (std::abs(det) < 1e-12) return {calib.R10_px, calib.delta_px};
    return {(Sr * Skk - Sk * Skr) / det, (S1 * Skr - Sk * Sr) / det};
}

} // namespace bestefar
