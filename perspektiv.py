"""
Perspektiv-retting (rektifisering) av skivebildet.

Ringfamilien er konsentriske sirkler i skiveplanet, men avbildes med
perspektiv: hver ring blir en ellipse hvis tilsynelatende senter driver
med radius. En homografi med 6 frie parametre tilpasses slik at alle
ringene blir konsentriske sirkler igjen:

    p' = p - c                       (sentrering)
    w  = 1 + l1*x' + l2*y'           (projektiv del - vanishing line)
    q  = (x'/w, y'/w)
    u  = q_x + k*q_y                 (skew)
    v  = (1+m)*q_y                   (akseforhold)

Parametre: c (2), l1, l2, k, m. Rotasjon/skala er irrelevant for sirkler
og holdes fast. Tilpasses med minste kvadrater over punkter samplet fra
ringenes harmoniske fit (k<=2 fanger perspektivavbildningen av en sirkel
til forste orden, og punktene er allerede outlier-renset).
"""

import cv2
import numpy as np
from scipy.optimize import least_squares


def _ring_points_from_harmonics(calib, n_theta=180):
    """
    Sample punkter (bildekoordinater) per inlier-ring fra harmonisk fit.

    Returns:
        liste av (value, pts) der pts er (n_theta, 2) float64
    """
    cx, cy = calib['center']
    theta = 2.0 * np.pi * np.arange(n_theta) / n_theta
    basis = np.column_stack([np.ones_like(theta), np.sin(theta), np.cos(theta),
                             np.sin(2 * theta), np.cos(2 * theta)])
    out = []
    for (value, beta, rmse, cov) in calib['ring_harmonics']:
        r = basis @ beta
        pts = np.column_stack([cx + r * np.cos(theta), cy + r * np.sin(theta)])
        out.append((value, pts))
    return out


def _apply_params(params, pts):
    """Anvend rektifiseringsparametre på punkter (N, 2). Returnerer (N, 2)."""
    dcx, dcy, l1, l2, k, m = params
    x = pts[:, 0] - dcx
    y = pts[:, 1] - dcy
    w = 1.0 + l1 * x + l2 * y
    qx = x / w
    qy = y / w
    u = qx + k * qy
    v = (1.0 + m) * qy
    return np.column_stack([u, v])


def fit_rectification(calib, cfg, debug_lines=None):
    """
    Tilpass rektifiseringsparametre slik at alle ringer blir konsentriske
    sirkler rundt origo (i sentrert koordinatsystem).

    Returns:
        (H, params, rms) der H er 3x3 homografi i originalkoordinater
        (original -> rektifisert, bevarer kalibreringssenteret), eller None.
    """
    ring_pts = _ring_points_from_harmonics(calib, cfg.get('persp_n_theta', 180))
    if len(ring_pts) < 3:
        return None

    cx, cy = calib['center']
    groups = [pts - np.array([cx, cy]) for (v, pts) in ring_pts]
    sizes = [len(g) for g in groups]

    def residuals(params):
        res = []
        for g in groups:
            q = _apply_params(params, g)
            r = np.hypot(q[:, 0], q[:, 1])
            res.append(r - np.mean(r))
        return np.concatenate(res)

    x0 = np.zeros(6)
    # l1/l2 er ~1/avstand til vanishing line: sma verdier, gi skala via x_scale
    sol = least_squares(residuals, x0, method='lm',
                        x_scale=[1.0, 1.0, 1e-5, 1e-5, 1e-2, 1e-2])
    if not sol.success and debug_lines is not None:
        debug_lines.append(f"perspektiv: least_squares ikke konvergert: {sol.message}")

    rms_before = float(np.sqrt(np.mean(residuals(x0) ** 2)))
    rms_after = float(np.sqrt(np.mean(sol.fun ** 2)))
    if debug_lines is not None:
        dcx, dcy, l1, l2, k, m = sol.x
        debug_lines.append(
            f"perspektiv: rms {rms_before:.2f} -> {rms_after:.2f}px, "
            f"dc=({dcx:.2f},{dcy:.2f}), l=({l1:.2e},{l2:.2e}), "
            f"skew={k:.4f}, akse={m:.4f}")

    # Bygg homografi i originalkoordinater:
    # T(-c-dc) -> projektiv -> affin -> T(+c)
    dcx, dcy, l1, l2, k, m = sol.x
    T1 = np.array([[1, 0, -(cx + dcx)], [0, 1, -(cy + dcy)], [0, 0, 1]], dtype=np.float64)
    P = np.array([[1, 0, 0], [0, 1, 0], [l1, l2, 1]], dtype=np.float64)
    A = np.array([[1, k, 0], [0, 1 + m, 0], [0, 0, 1]], dtype=np.float64)
    T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float64)
    H = T2 @ A @ P @ T1
    return H, sol.x, rms_after


def warp_image(img, H):
    """Rektifiser bildet med homografien (samme størrelse ut)."""
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)


def transform_points(pts, H):
    """Transformer (N, 2)-punkter med homografi."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, H).reshape(-1, 2)


def transform_points_inverse(pts, H):
    """Transformer (N, 2)-punkter med invers homografi."""
    return transform_points(pts, np.linalg.inv(H))
