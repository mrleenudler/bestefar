"""
Visualisering for C3: stemmekart, overlapp-kandidater, finale treff.
Output: Visualiseringer/outputs/C3_overlap_{votemap,cands,final}.png
"""
import cv2, numpy as np
from pathlib import Path
from config import DEFAULT_CONFIG
import inspect_hits as ih
import circles as circle_det
from overlap import _odd, _extract_patch, _crescent_mask

cfg = DEFAULT_CONFIG.copy()
OUT = Path('Visualiseringer/outputs')
OUT.mkdir(parents=True, exist_ok=True)

# ── last og preparer ──────────────────────────────────────────────────────────
img = cv2.imread('Testsett/C3.jpg')
got = ih.prep(img, 'C3')
img_d, gray_d, calib = got
cx0, cy0 = calib['center']
delta    = calib['delta_px']
R10      = calib['R10_px']
search_r = cfg['hit_search_r_max_frac'] * (R10 + 9.0 * delta)
marker_r = cfg['hit_marker_radius_frac'] * delta
dot_r    = cfg['hit_dot_radius_frac'] * delta

# ── hoveddetektor: 8 treff + stemmekart ──────────────────────────────────────
cands, dbg = circle_det.detect_circles(
    gray_d, (cx0, cy0), search_r, marker_r, dot_r, cfg, inner_r=0.0, return_dbg=True)
hits8 = [{'x': X, 'y': Y, 'score': sc} for X, Y, sc in cands]

acc  = dbg['acc']
ox, oy = dbg['offset']
amax = float(acc.max())
thr  = dbg['thr']

# ── 1. STEMMEKART ─────────────────────────────────────────────────────────────
# Skalert mot terskel: under terskel = blaa/kald, ekte topp = gul/roed
accv  = np.clip(acc / (thr + 1e-6), 0, 1.5) / 1.5
heat  = cv2.applyColorMap((accv * 255).astype(np.uint8), cv2.COLORMAP_JET)
heat  = cv2.resize(heat, (img_d.shape[1], img_d.shape[0]))

# Merk de 8 allerede-detekterte treffene
for h in hits8:
    pt = (int(round(h['x'])), int(round(h['y'])))
    cv2.circle(heat, pt, int(round(marker_r)), (255, 255, 255), 2)
    cv2.putText(heat, f"{h['score']:.2f}", (pt[0]+6, pt[1]-6),
                0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

# Marker overlapp-kandidat (1413,941) med gul pil
tgt = (1413, 941)
cv2.circle(heat, tgt, int(round(marker_r)), (0, 255, 255), 2)
cv2.arrowedLine(heat, (tgt[0]+80, tgt[1]-60), (tgt[0]+10, tgt[1]-10),
                (0, 255, 255), 2, tipLength=0.3)
cv2.putText(heat, "OVERLAPP", (tgt[0]+85, tgt[1]-65),
            0, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
cv2.imwrite(str(OUT / 'C3_overlap_votemap.png'), heat)
print(f'Stemmekart lagret  (amax={amax:.0f} thr={thr:.0f})')

# ── 2. OVERLAPP-KANDIDATER (subterskel, stoerste NMS) ────────────────────────
nms_r  = _odd(cfg['hit_overlap_nms_frac'] * marker_r)
vote_floor = cfg['hit_overlap_vote_frac'] * amax
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nms_r, nms_r))
dil    = cv2.dilate(acc, kernel)
py_all, px_all = np.where((acc == dil) & (acc >= vote_floor))

max_dist    = cfg['hit_overlap_max_dist_frac'] * marker_r
min_offset  = cfg['hit_overlap_min_offset_frac'] * marker_r
min_d_other = cfg['hit_min_dist_frac'] * marker_r
max_anc_r   = cfg['hit_overlap_max_anchor_r_frac'] * delta

anchors = [h for h in hits8 if np.hypot(h['x']-cx0, h['y']-cy0) <= max_anc_r]

# Finn alle kandidater som passerer avstandsfilter (foer NCC)
pre_ncc = []
for anchor in anchors:
    ax, ay = anchor['x'], anchor['y']
    for py, px in zip(py_all, px_all):
        X = float(px + ox); Y = float(py + oy)
        d = np.hypot(X-ax, Y-ay)
        if d < min_offset or d > max_dist:
            continue
        if any(np.hypot(X-k['x'], Y-k['y']) < min_d_other
               for k in hits8 if k is not anchor):
            continue
        if np.hypot(X-cx0, Y-cy0) > search_r:
            continue
        pre_ncc.append({'x': X, 'y': Y, 'v': float(acc[py, px]), 'anchor': anchor})

cands_img = img_d.copy()
# Alle 8 treff (hvite sirkler)
for h in hits8:
    pt = (int(round(h['x'])), int(round(h['y'])))
    cv2.circle(cands_img, pt, int(round(marker_r)), (200, 200, 200), 2)
# Sub-terskel peaks som passerte avstandsfilter (orange)
for c in pre_ncc:
    pt = (int(round(c['x'])), int(round(c['y'])))
    frac = c['v'] / amax
    cv2.circle(cands_img, pt, int(round(marker_r)), (0, 165, 255), 2)
    cv2.putText(cands_img, f"{frac:.2f}", (pt[0]+6, pt[1]-6),
                0, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
cv2.putText(cands_img, f"Overlapp-kandidater (foer NCC): {len(pre_ncc)}",
            (20, 35), 0, 0.9, (0, 165, 255), 2, cv2.LINE_AA)
cv2.imwrite(str(OUT / 'C3_overlap_cands.png'), cands_img)
print(f'Kandidat-kart lagret  ({len(pre_ncc)} kandidater foer NCC, nms={nms_r}px vote_floor={vote_floor:.1f})')

# ── 3. FINALE TREFF (etter NCC-validering) ───────────────────────────────────
# Gjenbruk crescent-NCC slik find_overlap_hits() gjoer det
def isolation(h):
    others = [o for o in hits8 if o is not h]
    return min(np.hypot(h['x']-o['x'], h['y']-o['y']) for o in others) * h['score'] if others else 0

src    = max(hits8, key=isolation)
r_lo   = cfg['hit_overlap_tmpl_r_lo'] * marker_r
r_hi   = cfg['hit_overlap_tmpl_r_hi'] * marker_r
tmpl_r_px = int(r_hi) + 2
tmpl_f = _extract_patch(gray_d, src['x'], src['y'], tmpl_r_px).astype(np.float32) / 255.0
ncc_thresh = cfg['hit_overlap_ncc_thresh']

accepted = []
for c in pre_ncc:
    X, Y = c['x'], c['y']
    ax, ay = c['anchor']['x'], c['anchor']['y']
    dom_dx = ax - X; dom_dy = ay - Y
    cmask   = _crescent_mask(tmpl_r_px, r_lo, r_hi, dom_dx, dom_dy, marker_r)
    cmask_f = (cmask > 0).astype(np.float32)
    if cmask_f.sum() < 10:
        continue
    cp  = _extract_patch(gray_d, X, Y, tmpl_r_px).astype(np.float32) / 255.0
    res = cv2.matchTemplate(cp, tmpl_f, cv2.TM_CCOEFF_NORMED, mask=cmask_f)
    ncc = float(res.max()) if res.size > 0 else 0.0
    c['ncc'] = ncc
    if ncc >= ncc_thresh:
        accepted.append(c)

final_img = img_d.copy()
# Originale 8 treff
for h in hits8:
    pt = (int(round(h['x'])), int(round(h['y'])))
    cv2.circle(final_img, pt, int(round(marker_r)), (0, 200, 0), 2)
# Nye overlapp-treff (cyan)
for c in accepted:
    pt = (int(round(c['x'])), int(round(c['y'])))
    cv2.circle(final_img, pt, int(round(marker_r)), (0, 255, 255), 3)
    cv2.putText(final_img, f"NCC={c['ncc']:.2f}", (pt[0]+6, pt[1]-6),
                0, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.arrowedLine(final_img, (pt[0]+90, pt[1]-70), (pt[0]+10, pt[1]-10),
                    (0, 255, 255), 2, tipLength=0.3)
cv2.putText(final_img, f"Finale treff: {len(hits8)+len(accepted)}/9  (+{len(accepted)} overlapp)",
            (20, 35), 0, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
cv2.imwrite(str(OUT / 'C3_overlap_final.png'), final_img)
print(f'Finale kart lagret  ({len(accepted)} overlapp-treff godkjent, NCC-terskel={ncc_thresh})')

for c in pre_ncc:
    ncc = c.get('ncc', float('nan'))
    print(f"  ({c['x']:.0f},{c['y']:.0f})  v={c['v']:.1f}({c['v']/amax:.2f})  "
          f"NCC={ncc:.3f}  => {'GODKJENT' if ncc >= ncc_thresh else 'AVVIST'}")
