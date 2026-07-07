import cv2
m = cv2.imread('Visualiseringer/outputs/Cset_screen_montage.png')
H = m.shape[0]
hdr = 40
row_h = (H - hdr) // 10
for idx in (2, 4, 10):
    y0 = hdr + (idx - 1) * row_h
    row = m[y0:y0 + row_h]
    # upscale x2 for readability
    row = cv2.resize(row, (row.shape[1] * 2, row.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f'Visualiseringer/outputs/_row_C{idx}.png', row)
    print(f'C{idx} -> _row_C{idx}.png  {row.shape[1]}x{row.shape[0]}')
