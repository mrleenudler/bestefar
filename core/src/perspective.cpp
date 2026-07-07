#include "perspective.h"

#include <cmath>
#include <vector>

namespace bestefar {
namespace {

// Residualmodell fra perspektiv.py: per ring, radius etter transformasjon
// minus ringens middelradius.
struct Model {
    std::vector<std::vector<cv::Point2d>> groups;   // sentrerte ringpunkter

    int n_resid() const {
        int n = 0;
        for (const auto& g : groups) n += static_cast<int>(g.size());
        return n;
    }

    void residuals(const double* p, double* out) const {
        const double dcx = p[0], dcy = p[1], l1 = p[2], l2 = p[3], k = p[4], m = p[5];
        int idx = 0;
        for (const auto& g : groups) {
            const int n = static_cast<int>(g.size());
            std::vector<double> r(n);
            double mean = 0;
            for (int i = 0; i < n; ++i) {
                const double x = g[i].x - dcx, y = g[i].y - dcy;
                const double w = 1.0 + l1 * x + l2 * y;
                const double qx = x / w, qy = y / w;
                const double u = qx + k * qy, v = (1.0 + m) * qy;
                r[i] = std::hypot(u, v);
                mean += r[i];
            }
            mean /= n;
            for (int i = 0; i < n; ++i) out[idx++] = r[i] - mean;
        }
    }
};

} // namespace

std::optional<cv::Mat> fit_rectification(const Calibration& calib, const Config& cfg) {
    if (calib.harmonics.size() < 3) return std::nullopt;
    const double cx = calib.center.x, cy = calib.center.y;
    const int n_theta = cfg.persp_n_theta;

    Model model;
    for (const auto& h : calib.harmonics) {
        std::vector<cv::Point2d> pts;
        pts.reserve(n_theta);
        for (int i = 0; i < n_theta; ++i) {
            const double th = 2.0 * CV_PI * i / n_theta;
            const double r = h.beta[0] + h.beta[1] * std::sin(th) + h.beta[2] * std::cos(th) +
                             h.beta[3] * std::sin(2 * th) + h.beta[4] * std::cos(2 * th);
            pts.emplace_back(r * std::cos(th), r * std::sin(th));  // allerede sentrert
        }
        model.groups.push_back(std::move(pts));
    }

    // Levenberg-Marquardt paa z, params = scale .* z (som scipy x_scale)
    const double scale[6] = {1.0, 1.0, 1e-5, 1e-5, 1e-2, 1e-2};
    double z[6] = {0, 0, 0, 0, 0, 0};
    const int nr = model.n_resid();
    std::vector<double> F(nr), F2(nr), Ftmp(nr);
    std::vector<std::vector<double>> J(6, std::vector<double>(nr));

    auto eval = [&](const double* zz, double* out) {
        double p[6];
        for (int i = 0; i < 6; ++i) p[i] = scale[i] * zz[i];
        model.residuals(p, out);
    };

    eval(z, F.data());
    double cost = 0;
    for (double f : F) cost += f * f;

    double lambda = 1e-3;
    for (int iter = 0; iter < 100; ++iter) {
        // Numerisk jacobian (forover-differens paa z)
        const double h = 1e-6;
        for (int j = 0; j < 6; ++j) {
            double zt[6];
            std::copy(z, z + 6, zt);
            zt[j] += h;
            eval(zt, Ftmp.data());
            for (int i = 0; i < nr; ++i) J[j][i] = (Ftmp[i] - F[i]) / h;
        }
        // Normalligninger
        cv::Mat A(6, 6, CV_64F), g(6, 1, CV_64F);
        for (int a = 0; a < 6; ++a) {
            double gv = 0;
            for (int i = 0; i < nr; ++i) gv += J[a][i] * F[i];
            g.at<double>(a) = -gv;
            for (int b = 0; b < 6; ++b) {
                double s = 0;
                for (int i = 0; i < nr; ++i) s += J[a][i] * J[b][i];
                A.at<double>(a, b) = s;
            }
        }
        bool improved = false;
        for (int tries = 0; tries < 10; ++tries) {
            cv::Mat Ad = A.clone();
            for (int d = 0; d < 6; ++d)
                Ad.at<double>(d, d) += lambda * std::max(A.at<double>(d, d), 1e-12);
            cv::Mat dz;
            if (!cv::solve(Ad, g, dz, cv::DECOMP_CHOLESKY) &&
                !cv::solve(Ad, g, dz, cv::DECOMP_SVD)) break;
            double z2[6];
            for (int i = 0; i < 6; ++i) z2[i] = z[i] + dz.at<double>(i);
            eval(z2, F2.data());
            double cost2 = 0;
            for (double f : F2) cost2 += f * f;
            if (cost2 < cost) {
                std::copy(z2, z2 + 6, z);
                F = F2;
                const double rel = (cost - cost2) / std::max(cost, 1e-30);
                cost = cost2;
                lambda = std::max(lambda * 0.3, 1e-12);
                improved = true;
                if (rel < 1e-10) iter = 100;   // konvergert
                break;
            }
            lambda *= 10.0;
        }
        if (!improved) break;
    }

    const double dcx = scale[0] * z[0], dcy = scale[1] * z[1];
    const double l1 = scale[2] * z[2], l2 = scale[3] * z[3];
    const double k = scale[4] * z[4], m = scale[5] * z[5];

    const cv::Mat T1 = (cv::Mat_<double>(3, 3) << 1, 0, -(cx + dcx), 0, 1, -(cy + dcy), 0, 0, 1);
    const cv::Mat P = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, l1, l2, 1);
    const cv::Mat Am = (cv::Mat_<double>(3, 3) << 1, k, 0, 0, 1 + m, 0, 0, 0, 1);
    const cv::Mat T2 = (cv::Mat_<double>(3, 3) << 1, 0, cx, 0, 1, cy, 0, 0, 1);
    return cv::Mat(T2 * Am * P * T1);
}

cv::Mat warp_image(const cv::Mat& img, const cv::Mat& H) {
    cv::Mat out;
    cv::warpPerspective(img, out, H, img.size(), cv::INTER_LINEAR);
    return out;
}

cv::Point2d transform_point(const cv::Point2d& p, const cv::Mat& H) {
    std::vector<cv::Point2d> in{p}, out;
    cv::perspectiveTransform(in, out, H);
    return out[0];
}

cv::Point2d transform_point_inverse(const cv::Point2d& p, const cv::Mat& H) {
    return transform_point(p, H.inv());
}

} // namespace bestefar
