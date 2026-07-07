// Poengberegning. Portert fra scoring.py.
#pragma once

#include <vector>

#include "bestefar/config.h"
#include "bestefar/types.h"

namespace bestefar {

struct ScorePart {
    double distance, theta, decimal;
    int integer;
};

ScorePart score_hit(double x, double y, const Calibration& calib, const Config& cfg);

} // namespace bestefar
