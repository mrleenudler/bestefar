// Perspektiv-rektifisering: 6-param modell (dc, l1, l2, k, m) tilpasses saa
// ringene blir konsentriske sirkler. Portert fra perspektiv.py; scipy
// least_squares(method='lm') erstattet av haandrullet Levenberg-Marquardt
// med samme diagonalskalering.
#pragma once

#include <opencv2/opencv.hpp>
#include <optional>

#include "bestefar/config.h"
#include "bestefar/types.h"

namespace bestefar {

// Returnerer 3x3 homografi original->rektifisert (bevarer kalibreringssenteret),
// eller nullopt hvis < 3 ringer.
std::optional<cv::Mat> fit_rectification(const Calibration& calib, const Config& cfg);

cv::Mat warp_image(const cv::Mat& img, const cv::Mat& H);
cv::Point2d transform_point(const cv::Point2d& p, const cv::Mat& H);
cv::Point2d transform_point_inverse(const cv::Point2d& p, const cv::Mat& H);

} // namespace bestefar
