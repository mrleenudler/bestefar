// Desktop-CLI: analyser et bilde, skriv JSON til stdout.
// Brukes av verify_cset_cpp.py for aa kjoere C-sett-oraklet mot C++-kjernen.
// --debug: skriv mellomsteg (skjermquad, ytre sirkel, kalibrering) til stderr.
#include <cstdio>
#include <cstring>
#include <opencv2/imgcodecs.hpp>

#include "bestefar/analyze.h"
#include "bestefar/bestefar_ffi.h"
#include "../src/outer_circle.h"
#include "../src/preprocess.h"
#include "../src/rings.h"
#include "../src/screen.h"

static void debug_stages(const cv::Mat& img, const bestefar::Config& cfg) {
    using namespace bestefar;
    auto screen = rectify_to_screen(img, cfg);
    if (!screen) { std::fprintf(stderr, "DBG screen: FEIL\n"); return; }
    std::fprintf(stderr, "DBG screen: crop %dx%d\n",
                 screen->warped.cols, screen->warped.rows);
    auto oc = detect_outer_circle(screen->warped, cfg);
    if (!oc) { std::fprintf(stderr, "DBG outer: FEIL\n"); return; }
    std::fprintf(stderr, "DBG outer: c=(%.1f,%.1f) r=%.1f\n", oc->cx, oc->cy, oc->r);
    cv::Mat gray = preprocess::to_gray(screen->warped);
    auto calib = calibrate_and_refine(gray, {oc->cx, oc->cy}, oc->r, cfg);
    if (!calib) { std::fprintf(stderr, "DBG rings: FEIL (nullopt)\n"); return; }
    std::fprintf(stderr, "DBG rings: delta=%.2f R10=%.2f n=%zu senter=(%.1f,%.1f)\n",
                 calib->delta_px, calib->R10_px, calib->harmonics.size(),
                 calib->center.x, calib->center.y);
    auto [ok, reason] = validate_calibration(*calib, cfg);
    std::fprintf(stderr, "DBG gate: %s %s\n", ok ? "OK" : "AVVIST", reason.c_str());
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "bruk: bestefar_cli <bilde.jpg> [--debug]\n");
        return 2;
    }
    if (std::strcmp(argv[1], "--votetest") == 0 && argc >= 5) {
        // bruk: bestefar_cli --votetest punkter.txt W H  (rader: x y ux uy mag)
        FILE* f = std::fopen(argv[2], "r");
        if (!f) return 2;
        bestefar::VoteTestPoints pts;
        double x, y, ux, uy, mag;
        while (std::fscanf(f, "%lf %lf %lf %lf %lf", &x, &y, &ux, &uy, &mag) == 5) {
            pts.x.push_back(static_cast<float>(x));
            pts.y.push_back(static_cast<float>(y));
            pts.ux.push_back(static_cast<float>(ux));
            pts.uy.push_back(static_cast<float>(uy));
            pts.mag.push_back(static_cast<float>(mag));
        }
        std::fclose(f);
        const int W = std::atoi(argv[3]), H = std::atoi(argv[4]);
        bestefar::Config cfg;
        const auto c = bestefar::vote_test(pts, cfg, H, W);
        std::printf("votetest c0=(%.2f,%.2f) n=%zu\n", c.x, c.y, pts.x.size());
        return 0;
    }
    const cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (img.empty()) {
        std::printf("{\"status\":\"ERROR_BAD_INPUT\",\"message\":\"kunne ikke lese bilde\"}\n");
        return 1;
    }
    if (argc >= 3 && std::strcmp(argv[2], "--debug") == 0) {
        bestefar::Config dcfg;
        debug_stages(img, dcfg);
    }
    if (argc >= 3 && std::strcmp(argv[2], "--probe") == 0) {
        // Kjoer FrameProbe paa et stillbilde: validerer auto-capture-gatene
        // mot ekte apparatfoto (C-settet) uten telefon.
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        BfImage bimg{gray.data, gray.cols, gray.rows,
                     static_cast<int32_t>(gray.step), BF_FMT_GRAY8};
        BfAutoCapture* ac = bf_autocapture_create(nullptr);
        BfFrameProbe probe;
        bf_autocapture_feed(ac, &bimg, &probe);
        bf_autocapture_destroy(ac);
        std::fprintf(stderr,
                     "PROBE roi=%d skarp=%.0f lo=%.3f hi=%.3f dek=%.2f "
                     "str=%.2f bull=%.2f kval=%d storrelse=%d\n",
                     probe.roi_found, probe.sharpness, probe.clip_lo_frac,
                     probe.clip_hi_frac, probe.coverage, probe.screen_width_frac,
                     probe.bull_width_frac, probe.quality_ok, probe.size_ok);
        return 0;
    }
    if (argc >= 3 && std::strcmp(argv[2], "--outeronly") == 0) {
        bestefar::Config dcfg;
        auto oc = bestefar::detect_outer_circle(img, dcfg);   // ingen skjermdeteksjon
        if (oc) std::fprintf(stderr, "OUTERONLY: c=(%.1f,%.1f) r=%.1f\n",
                             oc->cx, oc->cy, oc->r);
        else std::fprintf(stderr, "OUTERONLY: FEIL\n");
        return 0;
    }
    if (argc >= 4 && std::strcmp(argv[2], "--dumpcrop") == 0) {
        bestefar::Config dcfg;
        auto screen = bestefar::rectify_to_screen(img, dcfg);
        if (screen) {
            cv::imwrite(argv[3], screen->warped);
            std::fprintf(stderr, "DBG dump: %s (%dx%d)\n", argv[3],
                         screen->warped.cols, screen->warped.rows);
        }
        return 0;
    }

    bestefar::Config cfg;   // defaults = referanseverdiene
    const auto res = bestefar::analyze_target(img, cfg);

    const char* status_str = "ERROR_INTERNAL";
    switch (res.status) {
        case bestefar::Status::Ok: status_str = "OK"; break;
        case bestefar::Status::RejectedNoScreen: status_str = "REJECTED_NO_SCREEN"; break;
        case bestefar::Status::RejectedNoRings: status_str = "REJECTED_NO_RINGS"; break;
        case bestefar::Status::RejectedInvalidTarget: status_str = "REJECTED_INVALID_TARGET"; break;
        case bestefar::Status::RejectedNoHits: status_str = "REJECTED_NO_HITS"; break;
        case bestefar::Status::ErrorBadInput: status_str = "ERROR_BAD_INPUT"; break;
        default: break;
    }

    std::printf("{\"status\":\"%s\",\"message\":\"%s\",", status_str, res.message.c_str());
    std::printf("\"sum_decimal\":%.1f,\"sum_integer\":%d,", res.sum_decimal, res.sum_integer);
    std::printf("\"confidence\":%.3f,\"n_rings\":%d,", res.confidence.overall,
                res.confidence.n_rings);
    std::printf("\"hits\":[");
    for (size_t i = 0; i < res.hits.size(); ++i) {
        const auto& h = res.hits[i];
        std::printf("%s{\"x\":%.1f,\"y\":%.1f,\"r_rel\":%.3f,\"theta\":%.4f,"
                    "\"decimal\":%.1f,\"integer\":%d}",
                    i ? "," : "", h.x_orig, h.y_orig, h.r_rel, h.theta,
                    h.decimal, h.integer);
    }
    std::printf("]}\n");
    return res.status == bestefar::Status::Ok ? 0 : 1;
}
