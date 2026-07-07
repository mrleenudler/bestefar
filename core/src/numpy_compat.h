// Numeriske hjelpere med numpy-semantikk, slik at porten oppfoerer seg som
// Python-referansen. Kun det den porterte stien faktisk bruker.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace bestefar { namespace npc {

// np.percentile (default 'linear' interpolasjon). q i [0,100].
inline double percentile(std::vector<float> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double idx = (q / 100.0) * (static_cast<double>(v.size()) - 1.0);
    const size_t lo = static_cast<size_t>(std::floor(idx));
    const size_t hi = static_cast<size_t>(std::ceil(idx));
    if (lo == hi) return v[lo];
    const double frac = idx - static_cast<double>(lo);
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

// scipy.ndimage.gaussian_filter1d, mode='reflect' (default), truncate=4.0.
inline std::vector<double> gaussian_filter1d(const std::vector<double>& x, double sigma) {
    if (sigma <= 0 || x.empty()) return x;
    const int radius = static_cast<int>(4.0 * sigma + 0.5);
    std::vector<double> k(2 * radius + 1);
    double s = 0;
    for (int i = -radius; i <= radius; ++i) {
        k[i + radius] = std::exp(-0.5 * (i * i) / (sigma * sigma));
        s += k[i + radius];
    }
    for (auto& v : k) v /= s;
    const int n = static_cast<int>(x.size());
    std::vector<double> out(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double acc = 0;
        for (int j = -radius; j <= radius; ++j) {
            int idx = i + j;
            // reflect: (d c b a | a b c d | d c b a)
            while (idx < 0 || idx >= n) {
                if (idx < 0) idx = -idx - 1;
                if (idx >= n) idx = 2 * n - idx - 1;
            }
            acc += x[idx] * k[j + radius];
        }
        out[i] = acc;
    }
    return out;
}

// Enkle lokale maksima med hoeydeterskel (matcher find_peaks_1d-fallbacken i
// histogram.py: strengt stoerre enn begge naboer).
inline std::vector<int> local_maxima(const std::vector<double>& h, double min_height) {
    std::vector<int> peaks;
    for (int i = 1; i + 1 < static_cast<int>(h.size()); ++i)
        if (h[i] > h[i - 1] && h[i] > h[i + 1] && h[i] >= min_height)
            peaks.push_back(i);
    return peaks;
}

// Subpixel-parabel som rings._subpixel_parabola.
inline double subpixel_parabola(double vm, double v0, double vp) {
    const double denom = 2.0 * (vm - 2.0 * v0 + vp);
    if (std::abs(denom) < 1e-12) return 0.0;
    const double d = (vm - vp) / denom;
    if (!std::isfinite(d) || std::abs(d) > 0.5) return 0.0;
    return d;
}

}} // namespace bestefar::npc
