// Ringkalibrering og senterraffinering i polart domene. Portert fra rings.py
// (kun produksjonsstien: autokorr + progresjonsfit; comb_v2 er av og ikke med).
#pragma once

#include <opencv2/opencv.hpp>
#include <optional>
#include <string>

#include "bestefar/config.h"
#include "bestefar/types.h"

namespace bestefar {

std::optional<Calibration> calibrate_and_refine(const cv::Mat& gray, cv::Point2d center0,
                                                double r_outer_est, const Config& cfg,
                                                int refine_iters_override = -1);

// (ok, begrunnelse) — forkaster kalibreringer som ikke ligner en ekte skive.
std::pair<bool, std::string> validate_calibration(const Calibration& calib, const Config& cfg);

// R10/delta evaluert ved vinkel theta (perspektiv til foerste orden).
std::pair<double, double> local_ring_geometry(const Calibration& calib, double theta);

} // namespace bestefar
