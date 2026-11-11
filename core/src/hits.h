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

// noise_rejected (valgfri): settes true naar deteksjonen forkaster bildet som
// stoy (kandidat-/treffantall over hit_max_candidates/hit_max_hits) - da er
// returlisten tom og kalleren skal avvise bildet, ikke tolke det som "0 treff".
std::vector<Hit> detect_hits(const cv::Mat& gray, const Calibration& calib, const Config& cfg,
                             bool* noise_rejected = nullptr);

} // namespace bestefar
