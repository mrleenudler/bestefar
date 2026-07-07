// Portert fra preprocess.py.
#pragma once

#include <opencv2/opencv.hpp>

namespace bestefar { namespace preprocess {

// downscale_max_side: returnerer skalert bilde + scale (1.0 hvis ingen skalering).
inline std::pair<cv::Mat, double> downscale_max_side(const cv::Mat& img, int max_side) {
    const int max_dim = std::max(img.rows, img.cols);
    if (max_dim <= max_side) return {img.clone(), 1.0};
    const double scale = static_cast<double>(max_side) / max_dim;
    cv::Mat out;
    cv::resize(img, out, cv::Size(static_cast<int>(img.cols * scale),
                                  static_cast<int>(img.rows * scale)),
               0, 0, cv::INTER_AREA);
    return {out, scale};
}

inline cv::Mat to_gray(const cv::Mat& img) {
    if (img.channels() == 3) {
        cv::Mat g;
        cv::cvtColor(img, g, cv::COLOR_BGR2GRAY);
        return g;
    }
    return img.clone();
}

// gaussian_blur: Python bruker ksize = int(6*sigma+1) | 1 (eksplisitt kernel).
inline cv::Mat gaussian_blur(const cv::Mat& gray, double sigma) {
    int ksize = static_cast<int>(6 * sigma + 1);
    if (ksize % 2 == 0) ksize += 1;
    cv::Mat out;
    cv::GaussianBlur(gray, out, cv::Size(ksize, ksize), sigma);
    return out;
}

struct Gradients {
    cv::Mat gx, gy, mag, ux, uy;   // alle CV_32F
};

// compute_gradients: Scharr + normaliserte komponenter (eps som i Python).
inline Gradients compute_gradients(const cv::Mat& blur) {
    Gradients g;
    cv::Scharr(blur, g.gx, CV_32F, 1, 0);
    cv::Scharr(blur, g.gy, CV_32F, 0, 1);
    cv::magnitude(g.gx, g.gy, g.mag);
    cv::Mat mag_safe = g.mag + 1e-6f;
    cv::divide(g.gx, mag_safe, g.ux);
    cv::divide(g.gy, mag_safe, g.uy);
    return g;
}

// suppress_axis_normals: nullstill gradienter naer horisontal/vertikal.
inline cv::Mat suppress_axis_normals(const cv::Mat& ux, const cv::Mat& uy,
                                     const cv::Mat& mag, double thresh_deg) {
    cv::Mat out = mag.clone();
    if (thresh_deg <= 0) return out;
    const float t = static_cast<float>(std::tan(thresh_deg * CV_PI / 180.0));
    const float eps = 1e-6f;
    for (int y = 0; y < mag.rows; ++y) {
        const float* pux = ux.ptr<float>(y);
        const float* puy = uy.ptr<float>(y);
        float* po = out.ptr<float>(y);
        for (int x = 0; x < mag.cols; ++x) {
            const float aux = std::abs(pux[x]), auy = std::abs(puy[x]);
            const bool nearly_h = auy / (aux + eps) < t;
            const bool nearly_v = aux / (auy + eps) < t;
            if (nearly_h || nearly_v) po[x] = 0.0f;
        }
    }
    return out;
}

}} // namespace bestefar::preprocess
