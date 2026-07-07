#include "scoring.h"

#include <algorithm>
#include <cmath>

#include "rings.h"

namespace bestefar {

ScorePart score_hit(double x, double y, const Calibration& calib, const Config& cfg) {
    const double dx = x - calib.center.x, dy = y - calib.center.y;
    const double d = std::hypot(dx, dy);
    double theta = std::atan2(dy, dx);
    if (theta < 0) theta += 2.0 * CV_PI;

    double R10 = calib.R10_px, delta = calib.delta_px;
    if (cfg.score_use_local_rings) {
        auto [r10, dl] = local_ring_geometry(calib, theta);
        R10 = r10; delta = dl;
    }

    const double gauge = cfg.score_gauge_frac_of_delta * delta;
    double s = 10.0 + (R10 + gauge - d) / delta;
    s = std::clamp(s, 0.0, 10.9);

    double decimal;
    if (cfg.score_truncate)
        decimal = std::floor(s * 10.0 + 1e-9) / 10.0;   // offisiell avkorting
    else
        decimal = std::round(s * 10.0) / 10.0;
    const int integer = static_cast<int>(std::floor(decimal));
    return {d, theta, decimal, integer};
}

} // namespace bestefar
