import cv2
m = cv2.imread('Visualiseringer/outputs/Cset_mpi_square.png')
H, W = m.shape[:2]
# 5 rader x 2 kolonner. Separator 6px i midten.
row_h = H // 5
col_w = (W - 6) // 2
names = [['C1', 'C2'], ['C3', 'C4'], ['C5', 'C6'], ['C7', 'C8'], ['C9', 'C10']]
for r in range(5):
    for c in range(2):
        x0 = c * (col_w + 6)
        y0 = r * row_h
        panel = m[y0:y0 + row_h, x0:x0 + col_w]
        cv2.imwrite(f'Visualiseringer/outputs/_mpi_{names[r][c]}.png', panel)
print('OK')
