/* Ren C-FFI for Bestefar CV-kjernen (JNI/Swift). Ingen C++-typer, ingen
 * exceptions over grensen. Alle funksjoner er traadsikre saa lenge hver
 * traad bruker sitt eget BfAutoCapture-handle. */
#ifndef BESTEFAR_FFI_H
#define BESTEFAR_FFI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Statuskoder — hold i synk med bestefar::Status (types.h). */
#define BF_OK                      0
#define BF_REJECTED_NO_SCREEN      1
#define BF_REJECTED_NO_RINGS       2
#define BF_REJECTED_INVALID_TARGET 3
#define BF_REJECTED_NO_HITS        4
#define BF_ERROR_BAD_INPUT         100
#define BF_ERROR_INTERNAL          101

/* Pikselformater kjernen tar imot (plattformuavhengig inngang, kravspec §3). */
#define BF_FMT_GRAY8   0
#define BF_FMT_BGR8    1
#define BF_FMT_RGBA8   2
#define BF_FMT_NV21    3   /* Android YUV_420_888 pakket som NV21 */

typedef struct BfImage {
    const uint8_t* data;
    int32_t width;
    int32_t height;
    int32_t stride;      /* bytes per rad for plan 0 */
    int32_t format;      /* BF_FMT_* */
} BfImage;

typedef struct BfHit {
    double x_px, y_px;    /* i inputbildets koordinater */
    double r_rel;         /* radius i ringavstander fra kalibrert senter */
    double theta;         /* radianer */
    double decimal;       /* poeng med desimal (avkortet, offisiell regel) */
    int32_t integer;      /* heltallspoeng */
    double detect_score;
} BfHit;

#define BF_MAX_HITS 32

typedef struct BfResult {
    int32_t status;              /* BF_OK / BF_REJECTED_* / BF_ERROR_* */
    char    message[256];
    int32_t n_hits;
    BfHit   hits[BF_MAX_HITS];
    double  sum_decimal;
    int32_t sum_integer;
    double  confidence;          /* [0,1] interim (OCR-sjekk kommer senere) */
    int32_t n_rings;
    double  ring_resid_frac;
    int64_t timestamp_ms;        /* ekko av inngitt capture-tidsstempel */
} BfResult;

/* Analyser et stillbilde. timestamp_ms ekkos inn i resultatet (forskning §6).
 * Returnerer resultatets statuskode. */
int32_t bf_analyze(const BfImage* image, int64_t timestamp_ms, BfResult* out);

/* Kjernens egen versjon (semver-streng), UAVHENGIG av appens versionName —
 * se version.h. Statisk streng eid av kjernen; IKKE free() den. Formaalet er
 * aa vite hvilken kjerne som produserte en §6-donasjon (backend_spec.md §8),
 * ikke aa speile app-utgivelser. */
const char* bf_version(void);

/* ---- Auto-capture (kravspec §4) ------------------------------------- */
/* To adskilte kriterier: stabilitet og bildekvalitet. ALLE terskler er
 * UKALIBRERTE startverdier og skal kalibreres mot faktisk maskinvare. */

typedef struct BfAutoCaptureParams {
    int32_t stability_frames;        /* default 6 (~0.2 s holdevindu, capture-first v0.12) */
    double  stability_max_move_frac; /* default 0.01 (av diagonal) */
    double  min_sharpness;           /* default 40.0 (Laplacian-varians) */
    double  max_clip_lo_frac;        /* default 0.30 */
    double  max_clip_hi_frac;        /* default 0.20 */
    double  min_coverage;            /* default 0.80 */
    double  min_screen_width_frac;   /* default 0.90: skjermblobben maa fylle framen */
    double  min_bull_width_frac;     /* UKALIBRERT, default 0.10: bull-bredde/framebredde */
    double  max_glare_frac;          /* UKALIBRERT, default 0.05: gjenskinnsflekk/blob-areal */
    double  frame_margin_frac;       /* default 0.02 */
    int32_t probe_max_side;          /* default 480 */
} BfAutoCaptureParams;

typedef struct BfFrameProbe {
    int32_t roi_found;           /* apparat-ROI funnet i denne framen */
    double  quad[8];             /* TL,TR,BR,BL (x,y)-par i frame-koordinater */
    double  sharpness;           /* Laplacian-varians i ROI */
    double  clip_lo_frac;        /* andel piksler i moerkeste bin (undereksponert) */
    double  clip_hi_frac;        /* andel piksler i lyseste bin (utbrent) */
    double  coverage;            /* 1.0 = hele ROI godt innenfor framen */
    int32_t stable;              /* stabilitetskriteriet oppfylt naa */
    int32_t quality_ok;          /* kvalitetskriteriet oppfylt naa */
    int32_t should_capture;      /* begge kriterier oppfylt over vinduet -> knips! */
    double  screen_width_frac;   /* skjermblob-bredde / framebredde (stoerrelsesmaal) */
    int32_t size_ok;             /* stoerrelseskriteriet oppfylt naa (skjerm OG bull) */
    double  bull_width_frac;     /* bull-bredde / framebredde (presisjonsmaal) */
    double  glare_frac;          /* stoerste mettede flekk / skjermblob-areal */
    int32_t glare_ok;            /* gjenskinnskriteriet oppfylt naa */
} BfFrameProbe;

typedef struct BfAutoCapture BfAutoCapture;   /* opaque tilstandsmaskin */

void bf_autocapture_default_params(BfAutoCaptureParams* out);
BfAutoCapture* bf_autocapture_create(const BfAutoCaptureParams* params);
void bf_autocapture_destroy(BfAutoCapture* ac);
void bf_autocapture_reset(BfAutoCapture* ac);

/* Mat inn en live-frame (nedskaleres internt). Fyller ut probe.
 * Kall paa kamera-traaden; ~10 ms budsjett paa moderne mobil. */
int32_t bf_autocapture_feed(BfAutoCapture* ac, const BfImage* frame, BfFrameProbe* out);

#ifdef __cplusplus
}
#endif

#endif /* BESTEFAR_FFI_H */
