// Felles datatyper for kjernen. Speiler Python-pipelinens dict-strukturer.
#pragma once

#include <opencv2/core.hpp>
#include <cstdint>
#include <string>
#include <vector>

namespace bestefar {

// Python: ValueError-meldingene i _analyze_core, mappet til enum (ingen
// exceptions over FFI). VerdiER dokumentert i bestefar_ffi.h — hold i synk.
enum class Status : int32_t {
    Ok                    = 0,
    RejectedNoScreen      = 1,  // skjermdeteksjon feilet (og fallback av)
    RejectedNoRings       = 2,  // "fant ingen poengringer"
    RejectedInvalidTarget = 3,  // validate_calibration avviste (delta/ringer/resid)
    RejectedNoHits        = 4,  // "ingen treff funnet paa skiva"
    ErrorBadInput         = 100,
    ErrorInternal         = 101,
};

// Én ring i kalibreringen: verdi (1..10), harmonisk radius-modell
// r(theta) = beta0 + beta1*sin + beta2*cos + beta3*sin2 + beta4*cos2 (originalpx).
struct RingHarmonic {
    int    value;
    double beta[5];
    double rmse;
    double coverage;
};

// rings.calibrate_and_refine-resultatet.
struct Calibration {
    cv::Point2d center;
    double delta_px  = 0.0;
    double R10_px    = 0.0;
    std::vector<int>    ring_values;
    std::vector<double> ring_radii_px;
    std::vector<RingHarmonic> harmonics;
    double r_max = 0.0, rho_scale = 1.0;
};

// Ett detektert treff (analyse-ramme) + poeng.
struct ScoredHit {
    // Posisjon i originalfotoets koordinater:
    double x_orig = 0, y_orig = 0;
    // Relativ polar (kravspec §3): radius i ringavstander fra kalibrert senter.
    double r_rel = 0, theta = 0;
    double decimal = 0;
    int    integer = 0;
    double detect_score = 0;      // detektorens egen score (votemap/NCC)
    char   type = 'f';            // 'f' filled / 'o' outline
};

// Interim konfidens (kravspec §3-merknad: OCR-sjekk kommer senere).
struct Confidence {
    double overall = 0.0;         // [0,1]
    int    n_rings = 0;
    double ring_resid_frac = 0.0; // maks avvik fra ekvidistans (andel av delta)
    double mean_hit_score  = 0.0;
    bool   screen_used     = false;
};

struct AnalyzeResult {
    Status status = Status::ErrorInternal;
    std::string message;          // menneskelesbar aarsak ved avvisning
    std::vector<ScoredHit> hits;
    double sum_decimal = 0.0;
    int    sum_integer = 0;
    Confidence confidence;
    int64_t timestamp_ms = 0;     // ekko av capture-tidsstempel (forskning, §6)
    // aux-utvidelsespunkt: MPI-firkant o.l. kan legges til her senere uten
    // aa endre eksisterende felter (se docs/ARCHITECTURE.md).
};

} // namespace bestefar
