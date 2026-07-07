// Skjermdeteksjon og perspektiv-crop. Portert fra screen.py (inkl.
// kant-snapping fra 2026-07-02 som ga 10/10 paa C-settet).
#pragma once

#include <opencv2/opencv.hpp>
#include <optional>

#include "bestefar/config.h"

namespace bestefar {

struct ScreenRect {
    cv::Mat warped;              // perspektivkorrigert skjermutklipp (BGR)
    cv::Mat M;                   // 3x3 original -> warped
    std::array<cv::Point2f, 4> quad;  // TL,TR,BR,BL i originalkoordinater
};

std::optional<ScreenRect> rectify_to_screen(const cv::Mat& img_bgr, const Config& cfg);

// Kontrast-ROI alene — gjenbrukes av auto-capture (FrameProbe).
// gray maa vaere normalisert (percentil-strukket) arbeidsbilde.
cv::Mat apparatus_roi(const cv::Mat& gray_norm, const Config& cfg);
cv::Mat normalize_stretch(const cv::Mat& gray, const Config& cfg);

} // namespace bestefar
