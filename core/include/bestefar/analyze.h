// Hovedinngang for CV-kjernen. Portert fra Bestefar.py analyze_target/_analyze_core.
#pragma once

#include <opencv2/core.hpp>
#include <cstdint>

#include "bestefar/config.h"
#include "bestefar/types.h"

namespace bestefar {

// Analyserer et stillbilde av skiva. Kaster aldri; avvisninger rapporteres i
// result.status/message. timestamp_ms ekkos inn i resultatet (forskningskrav).
AnalyzeResult analyze_target(const cv::Mat& img_bgr, const Config& cfg,
                             int64_t timestamp_ms = 0);

} // namespace bestefar
