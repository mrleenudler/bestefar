"""Kjoer overlapp-pass paa C3, vis antall treff og poeng."""
import cv2
from config import DEFAULT_CONFIG
from Bestefar import analyze_target

cfg = DEFAULT_CONFIG.copy()
img = cv2.imread('Testsett/C3.jpg')
result = analyze_target(img, cfg=cfg)
hits = result.get('hits', [])
print(f'C3: {len(hits)} treff')
for h in sorted(hits, key=lambda x: x.get('score', 0), reverse=True):
    print(f'  ({h["x"]:.0f},{h["y"]:.0f})  score={h.get("score",0):.3f}  poeng={h.get("points", "?")}')
scores = [h.get('points', 0) for h in hits if h.get('points') is not None]
if scores:
    print(f'Sum={sum(scores):.1f}  Snitt={sum(scores)/len(scores):.2f}')
