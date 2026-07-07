// Ytre-sirkel-deteksjon via gradient-normal center voting.
// Portert fra Bestefar.py detect_outer_circle + voting.py/histogram.py/points.py.
// NMS/hysterese/radial-varians-raffinering er IKKE med (av i produksjon).
#pragma once

#include <opencv2/opencv.hpp>
#include <optional>
#include <vector>

#include "bestefar/config.h"

namespace bestefar {

struct OuterCircle {
    double cx = 0, cy = 0, r = 0;   // originalkoordinater
};

// Kaster ikke: nullopt ved "ingen edge points" (Python: ValueError).
std::optional<OuterCircle> detect_outer_circle(const cv::Mat& img_bgr, const Config& cfg);

// Test-inngang: kjoer pass1-stemmingen paa eksternt gitte punkter (feilsoking).
struct VoteTestPoints {
    std::vector<float> x, y, ux, uy, mag;
};
cv::Point2d vote_test(const VoteTestPoints& pts, const Config& cfg, int H, int W);

} // namespace bestefar
