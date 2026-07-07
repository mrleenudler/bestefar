// Treffdeteksjon. Portert fra hits.py — KUN enhetlig detektor-sti
// (hit_unified_detector=True i produksjon; Hough-stien er legacy).
#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "bestefar/config.h"
#include "bestefar/types.h"

namespace bestefar {

struct Hit {
    double x, y;
    char type;       // 'f' filled / 'o' outline
    double score;
};

std::vector<Hit> detect_hits(const cv::Mat& gray, const Calibration& calib, const Config& cfg);

} // namespace bestefar
