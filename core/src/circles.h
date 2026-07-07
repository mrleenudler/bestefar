// Stemmekart-detektor for treffmarkoerer. Portert fra circles.py.
#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "bestefar/config.h"

namespace bestefar {

struct CircleCand { double x, y, score; };

// Stemmekart i ROI-koordinater + offset. Brukes ogsaa av overlap-passet.
struct VoteMap {
    cv::Mat acc;
    cv::Point offset;
};

VoteMap circle_vote_map(const cv::Mat& gray, cv::Point2d center, double search_r,
                        double marker_r, double dot_r, const Config& cfg,
                        double inner_r = 0.0);

std::vector<CircleCand> detect_circles(const cv::Mat& gray, cv::Point2d center,
                                       double search_r, double marker_r, double dot_r,
                                       const Config& cfg, double inner_r = 0.0);

} // namespace bestefar
