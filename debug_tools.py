"""
Debug tools for visualization and logging (development only).
"""

import cv2
import numpy as np
from pathlib import Path
import time
from datetime import datetime


# Global variabel for å holde styr på visualiseringsindeks
visualization_index = 0

# Global buffer for å samle logg-oppføringer
_log_buffer = []

# Global variabel for å holde styr på total tid brukt i loggde operasjoner (i ms)
_logged_time_ms = 0.0


def save_visualization(image, title, index=None, visualization_dir=None):
    """
    Lagrer en visualisering som bilde i Visualiseringer katalogen.
    Bruker cv2.imwrite for ytelse.
    
    Args:
        image: Bildet å lagre (BGR eller grayscale)
        title: Beskrivende tittel
        index: Indeks for rekkefølge (hvis None, brukes global index)
        visualization_dir: Path til visualiseringskatalog (hvis None, brukes default)
    
    Returns:
        filepath: Path til lagret fil
    """
    global visualization_index
    
    if visualization_dir is None:
        visualization_dir = Path("Visualiseringer")
    
    # Opprett katalog hvis den ikke eksisterer
    visualization_dir.mkdir(exist_ok=True)
    
    # Bruk global index hvis ikke spesifisert
    if index is None:
        index = visualization_index
        visualization_index += 1
    
    # Formater tittel for filnavn (fjern spesialtegn)
    safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title)
    safe_title = safe_title.replace(' ', '_')
    
    # Lag filnavn med indeks og tittel
    filename = f"{index:03d}_{safe_title}.png"
    filepath = visualization_dir / filename
    
    # Konverter til uint8 hvis nødvendig
    if image.dtype != np.uint8:
        if image.dtype == np.float32 or image.dtype == np.float64:
            # Normaliser til [0, 255]
            if image.max() > 1.0:
                image = (image / image.max() * 255).astype(np.uint8)
            else:
                image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Lagre bildet
    cv2.imwrite(str(filepath), image)
    
    return filepath


def log_operation_time(filename, method_name, operation_name, execution_time):
    """
    Logger kjøretid for en operasjon.
    
    Args:
        filename: Navn på bildet som ble prosessert
        method_name: Navn på metoden/funksjonen
        operation_name: Navn på operasjonen
        execution_time: Kjøretid i sekunder
    """
    global _log_buffer, _logged_time_ms
    
    time_ms = execution_time * 1000.0
    
    # Logg kun hvis >= 1ms
    if time_ms >= 1.0:
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f'"{timestamp}" | {filename} | {method_name} | {operation_name} | {time_ms:.6f}'.replace('.', ',')
        _log_buffer.append(log_entry)
        _logged_time_ms += time_ms


def flush_log_buffer(log_file_path):
    """
    Skriver alle buffrede logg-oppføringer til fil.
    
    Args:
        log_file_path: Path til loggfil
    """
    global _log_buffer
    
    if len(_log_buffer) == 0:
        return
    
    mode = 'a'  # Append mode
    if not Path(log_file_path).exists():
        mode = 'w'  # Write mode for first write
    
    with open(log_file_path, mode, encoding='utf-8') as f:
        for entry in _log_buffer:
            f.write(entry + '\n')
    
    _log_buffer.clear()


def log_unaccounted_time(filename, total_time_ms, logged_time_ms):
    """
    Logger uforklart tid (total tid minus logget tid).
    
    Args:
        filename: Navn på bildet
        total_time_ms: Total kjøretid i ms
        logged_time_ms: Sum av logget tid i ms
    """
    unaccounted = total_time_ms - logged_time_ms
    if unaccounted > 0:
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f'"{timestamp}" | {filename} | program | uforklart_tid | {unaccounted:.6f}'.replace('.', ',')
        
        log_file = "ytelse.txt"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')


def create_pointset_overlay(gray_image, points_xy, color=(0, 255, 0)):
    """
    Create overlay image showing points on grayscale image.
    Uses direct pixel setting (1 pixel per point) for thin lines.
    
    Args:
        gray_image: Grayscale image
        points_xy: Dict with 'x', 'y' (numpy arrays)
        color: BGR color for points
    
    Returns:
        overlay: BGR image with points overlaid
    """
    if len(gray_image.shape) == 2:
        overlay = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = gray_image.copy()
    
    x = points_xy['x']
    y = points_xy['y']
    
    # Create mask from points (direct pixel setting, 1 pixel per point)
    h, w = gray_image.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    y_coords = y.astype(int)
    x_coords = x.astype(int)
    # Filter valid coordinates
    valid = (x_coords >= 0) & (x_coords < w) & (y_coords >= 0) & (y_coords < h)
    mask[y_coords[valid], x_coords[valid]] = True
    # Set pixels directly
    overlay[mask] = color
    
    return overlay


def reset_visualization_index():
    """Reset global visualization index."""
    global visualization_index
    visualization_index = 0

