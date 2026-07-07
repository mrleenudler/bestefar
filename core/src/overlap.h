// Delvis-skjulte treff: subterskel-stemmekart + maanesigd-NCC, og
// sentrum-sveip for radialt undertrykte treff. Portert fra overlap.py.
#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "bestefar/config.h"
#include "bestefar/types.h"

namespace bestefar {

struct Hit;   // hits.h

std::vector<Hit> find_overlap_hits(const cv::Mat& gray, const std::vector<Hit>& hits,
                                   const Calibration& calib, const Config& cfg);
std::vector<Hit> find_center_hits(const cv::Mat& gray, const std::vector<Hit>& hits,
                                  const Calibration& calib, const Config& cfg);

} // namespace bestefar
