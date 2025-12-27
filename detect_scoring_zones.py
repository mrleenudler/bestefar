"""
Computer Vision modul for å detektere poengsoner på skyteskive.
Dette er en prototype i Python/OpenCV som senere skal implementeres i C++.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from datetime import datetime
import os


def get_config():
    """
    Returnerer en dictionary med alle konstanter og input-variabler.
    Endre verdiene her for å justere oppførselen til algoritmene.
    
    Returns:
        dict: Dictionary med alle konfigurasjonsparametere
    """
    return {
        # Filer og kataloger
        'visualization_dir': Path("Visualiseringer"),
        'log_file_kronologisk': "ytelse_kronologisk.txt",
        'log_file_alfabetisk': "ytelse_alfabetisk.txt",
        
        # Bildeprosessering - Threshold
        'threshold_value': 127,           # Threshold-verdi for binær konvertering
        'threshold_max': 255,              # Maksimum verdi for threshold
        'threshold_type': cv2.THRESH_BINARY_INV,  # Threshold-type
        
        # Contour-deteksjon
        'contour_mode_retr_external': cv2.RETR_EXTERNAL,  # For find_outer_contour
        'contour_mode_retr_tree': cv2.RETR_TREE,          # For detect_scoring_zones_tree
        'contour_approx': cv2.CHAIN_APPROX_SIMPLE,        # Contour-approksimasjon
        
        # Contour-filtrering
        'min_contour_area': 100,          # Minimum areal for å beholde contour
        'circularity_threshold': 0.5,      # Minimum circularity for å beholde contour
        
        # Visualisering - Farger (BGR format)
        'color_green': (0, 255, 0),       # Grønn for contours/sirkler
        'color_red': (0, 0, 255),         # Rød for sentrum
        'color_blue': (255, 0, 0),        # Blå for tekst
        'color_magenta': (255, 0, 255),   # Magenta for hoved-sentrum
        
        # Visualisering - Tegneparametere
        'circle_thickness': 2,            # Tykkelse på sirkler
        'circle_center_size': 5,           # Størrelse på sentrum-markør
        'contour_thickness': 2,           # Tykkelse på contours
        'contour_thickness_large': 3,     # Tykkelse på største contour
        
        # Visualisering - Tekst
        'font': cv2.FONT_HERSHEY_SIMPLEX,
        'font_scale_small': 0.5,          # Font-størrelse for nummerering
        'font_scale_medium': 0.7,         # Font-størrelse for radius-tekst
        'font_thickness': 2,              # Font-tykkelse
        
        # Matplotlib
        'figure_size': (10, 10),          # Størrelse på matplotlib-figurer
        'save_dpi': 150,                   # DPI for lagrede bilder
        
        # Test-bilde (i main)
        'test_image_path': Path("Blink 1 (fake).jpg"),
    }


# Global variabel for å holde styr på visualiseringsindeks
visualization_index = 0

# Global buffer for å samle logg-oppføringer (for optimalisering)
_log_buffer = []

# Last inn konfigurasjon
config = get_config()


def save_visualization(image, title, index=None):
    """
    Lagrer en visualisering som bilde i Visualiseringer katalogen.
    
    Args:
        image: Bildet å lagre (BGR eller RGB)
        title: Beskrivende tittel
        index: Indeks for rekkefølge (hvis None, brukes global index)
    """
    global visualization_index, config
    
    # Opprett katalog hvis den ikke eksisterer
    config['visualization_dir'].mkdir(exist_ok=True)
    
    # Bruk global index hvis ikke spesifisert
    if index is None:
        index = visualization_index
        visualization_index += 1
    
    # Formater tittel for filnavn (fjern spesialtegn)
    safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title)
    safe_title = safe_title.replace(' ', '_')
    
    # Lag filnavn med indeks og tittel
    filename = f"{index:03d}_{safe_title}.png"
    filepath = config['visualization_dir'] / filename
    
    # Konverter til RGB hvis nødvendig (matplotlib forventer RGB)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Sjekk om det er BGR (OpenCV) eller RGB
        # Vi antar BGR hvis det kommer fra OpenCV
        if isinstance(image, np.ndarray) and image.dtype == np.uint8:
            # Konverter BGR til RGB for lagring
            start_time = time.time()
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            log_operation_time("visualization", "save_visualization", "cv2.cvtColor", time.time() - start_time)
        else:
            image_rgb = image
    else:
        image_rgb = image
    
    # Lagre bildet
    start_time = time.time()
    if len(image_rgb.shape) == 2:  # Gråskala
        plt.imsave(filepath, image_rgb, cmap='gray')
    else:  # Farge
        plt.imsave(filepath, image_rgb)
    log_operation_time("visualization", "save_visualization", "plt.imsave", time.time() - start_time)
    
    print(f"Lagret visualisering: {filepath}")
    return filepath


def log_performance(filename, method_name, algorithm, execution_time):
    """
    Logger ytelsesdata til to filer: kronologisk og alfabetisk.
    Bruker buffer for å unngå å skrive til fil for hver operasjon.
    
    Args:
        filename: Navn på bildet som ble prosessert
        method_name: Navn på metoden/funksjonen
        algorithm: Navn på algoritmen
        execution_time: Kjøretid i sekunder
    """
    global _log_buffer
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Konverter til millisekunder og bruk komma som desimaltegn
    time_ms = execution_time * 1000
    time_str = f"{time_ms:.6f}".replace('.', ',')
    entry = f'"{timestamp}" | {filename} | {method_name} | {algorithm} | {time_str}\n'
    
    # Legg til i buffer i stedet for å skrive direkte
    _log_buffer.append(entry)


def log_operation_time(filename, method_name, operation_name, execution_time):
    """
    Logger kjøretid for en spesifikk operasjon (f.eks. cv2-funksjon).
    Bruker buffer for å unngå å skrive til fil for hver operasjon.
    
    Args:
        filename: Navn på bildet som ble prosessert
        method_name: Navn på metoden/funksjonen som kaller operasjonen
        operation_name: Navn på operasjonen (f.eks. "cv2.imread", "cv2.cvtColor")
        execution_time: Kjøretid i sekunder
    """
    global _log_buffer
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Konverter til millisekunder og bruk komma som desimaltegn
    time_ms = execution_time * 1000
    time_str = f"{time_ms:.6f}".replace('.', ',')
    entry = f'"{timestamp}" | {filename} | {method_name} | {operation_name} | {time_str}\n'
    
    # Legg til i buffer i stedet for å skrive direkte
    _log_buffer.append(entry)


def flush_log_buffer():
    """
    Skriver alle buffrede logg-oppføringer til filene.
    Dette bør kalles når en funksjon/metode er ferdig.
    """
    global _log_buffer, config
    if not _log_buffer:
        return
    
    # Skriv alle til kronologisk fil (append)
    with open(config['log_file_kronologisk'], "a", encoding="utf-8") as f:
        f.writelines(_log_buffer)
    
    # For alfabetisk fil: les eksisterende, legg til nye, sorter, skriv
    entries = []
    if os.path.exists(config['log_file_alfabetisk']):
        with open(config['log_file_alfabetisk'], "r", encoding="utf-8") as f:
            entries = f.readlines()
    
    entries.extend(_log_buffer)
    entries.sort()
    
    with open(config['log_file_alfabetisk'], "w", encoding="utf-8") as f:
        f.writelines(entries)
    
    # Tøm buffer
    _log_buffer = []




def time_operation(filename, method_name, operation_name, func, *args, **kwargs):
    """
    Utfører en operasjon og logger kjøretiden.
    
    Args:
        filename: Navn på bildet som ble prosessert
        method_name: Navn på metoden/funksjonen som kaller operasjonen
        operation_name: Navn på operasjonen (f.eks. "cv2.imread")
        func: Funksjonen å kjøre
        *args, **kwargs: Argumenter til funksjonen
    
    Returns:
        Resultatet av funksjonen
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    log_operation_time(filename, method_name, operation_name, execution_time)
    return result


def find_outer_contour(image_path, visualize=True):
    """
    Finner den ytterste konturen ved å bruke minEnclosingCircle på den største konturen.
    
    Args:
        image_path: Sti til bildet
        visualize: Om resultatet skal visualiseres
    
    Returns:
        circle: (x, y, radius) for den ytterste sirkelen, eller None
        center: Sentrum (x, y)
    """
    start_time = time.time()
    filename = Path(image_path).name
    
    # Les bildet
    img = time_operation(filename, "find_outer_contour", "cv2.imread", 
                        cv2.imread, str(image_path))
    if img is None:
        raise ValueError(f"Kunne ikke lese bildet: {image_path}")
    
    save_visualization(img, "01_Originalt_bilde")
    
    # Konverter til gråskala
    gray = time_operation(filename, "find_outer_contour", "cv2.cvtColor", 
                         cv2.cvtColor, img, cv2.COLOR_BGR2GRAY)
    save_visualization(gray, "02_Graaskala", 2)
    
    # Threshold for å få binær bilde
    _, thresh = time_operation(filename, "find_outer_contour", "cv2.threshold", 
                              cv2.threshold, gray, config['threshold_value'], 
                              config['threshold_max'], config['threshold_type'])
    save_visualization(thresh, "03_Threshold_binar", 3)
    
    # Finn contours med RETR_EXTERNAL (kun ytterste contours)
    contours, hierarchy = time_operation(filename, "find_outer_contour", "cv2.findContours", 
                                        cv2.findContours, thresh, config['contour_mode_retr_external'], 
                                        config['contour_approx'])
    
    # Visualiser alle contours
    img_contours = img.copy()
    time_operation(filename, "find_outer_contour", "cv2.drawContours", 
                  cv2.drawContours, img_contours, contours, -1, config['color_green'], 
                  config['contour_thickness'])
    save_visualization(img_contours, "04_Alle_contours_RETR_EXTERNAL", 4)
    
    # Finn den største konturen
    if not contours:
        execution_time = time.time() - start_time
        log_performance(filename, "find_outer_contour", "RETR_EXTERNAL", execution_time)
        flush_log_buffer()
        return None, None
    
    # Time max() operasjon med contourArea
    start_max = time.time()
    # Beregn alle areas først for å time dem
    areas = []
    for c in contours:
        area = time_operation(filename, "find_outer_contour", "cv2.contourArea", 
                             cv2.contourArea, c)
        areas.append((area, c))
    largest_contour = max(areas, key=lambda x: x[0])[1]
    log_operation_time(filename, "find_outer_contour", "max(contours, key=contourArea)", time.time() - start_max)
    
    # Visualiser største contour
    img_largest = img.copy()
    time_operation(filename, "find_outer_contour", "cv2.drawContours", 
                  cv2.drawContours, img_largest, [largest_contour], -1, config['color_green'], 
                  config['contour_thickness_large'])
    save_visualization(img_largest, "05_Storste_contour", 5)
    
    # Bruk minEnclosingCircle på den største konturen
    (x, y), radius = time_operation(filename, "find_outer_contour", "cv2.minEnclosingCircle", 
                                    cv2.minEnclosingCircle, largest_contour)
    center = (int(x), int(y))
    radius = int(radius)
    
    # Visualiser resultatet
    result = img.copy()
    time_operation(filename, "find_outer_contour", "cv2.circle", 
                  cv2.circle, result, center, radius, config['color_green'], 
                  config['circle_thickness'])
    time_operation(filename, "find_outer_contour", "cv2.circle", 
                  cv2.circle, result, center, config['circle_center_size'], config['color_red'], -1)
    time_operation(filename, "find_outer_contour", "cv2.putText", 
                  cv2.putText, result, f"Radius: {radius}", (center[0] - 50, center[1] - radius - 20),
                  config['font'], config['font_scale_medium'], config['color_blue'], config['font_thickness'])
    save_visualization(result, "06_Resultat_find_outer_contour", 6)
    
    execution_time = time.time() - start_time
    log_performance(filename, "find_outer_contour", "RETR_EXTERNAL", execution_time)
    
    # Skriv alle logg-oppføringer til fil
    flush_log_buffer()
    
    if visualize:
        plt.figure(figsize=config['figure_size'])
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        plt.imshow(result_rgb)
        plt.title(f'Ytterste sirkel funnet (radius: {radius})')
        plt.axis('off')
        plt.show()
    
    return (center[0], center[1], radius), center


def detect_scoring_zones_tree(image_path, visualize=True):
    """
    Detekterer poengsoner ved å bruke RETR_TREE for å finne alle contours
    (inkludert nested contours for de koncentriske sirklene).
    
    Args:
        image_path: Sti til bildet
        visualize: Om resultatet skal visualiseres
    
    Returns:
        circles: Liste med detekterte sirkler (x, y, radius)
        center: Sentrum av skyteskiven (x, y)
    """
    start_time = time.time()
    filename = Path(image_path).name
    
    # Les bildet
    img = time_operation(filename, "detect_scoring_zones_tree", "cv2.imread", 
                        cv2.imread, str(image_path))
    if img is None:
        raise ValueError(f"Kunne ikke lese bildet: {image_path}")
    
    save_visualization(img, "01_Originalt_bilde")
    
    # Konverter til gråskala
    gray = time_operation(filename, "detect_scoring_zones_tree", "cv2.cvtColor", 
                         cv2.cvtColor, img, cv2.COLOR_BGR2GRAY)
    save_visualization(gray, "02_Graaskala", 2)
    
    # Threshold for å få binær bilde
    _, thresh = time_operation(filename, "detect_scoring_zones_tree", "cv2.threshold", 
                              cv2.threshold, gray, config['threshold_value'], 
                              config['threshold_max'], config['threshold_type'])
    save_visualization(thresh, "03_Threshold_binar", 3)
    
    # Finn contours med RETR_TREE (alle contours, inkludert nested)
    contours, hierarchy = time_operation(filename, "detect_scoring_zones_tree", "cv2.findContours", 
                                        cv2.findContours, thresh, config['contour_mode_retr_tree'], 
                                        config['contour_approx'])
    
    # Visualiser alle contours
    img_contours = img.copy()
    time_operation(filename, "detect_scoring_zones_tree", "cv2.drawContours", 
                  cv2.drawContours, img_contours, contours, -1, config['color_green'], 
                  config['contour_thickness'])
    save_visualization(img_contours, "04_Alle_contours_RETR_TREE", 4)
    
    # Filtrer contours basert på sirkel-likhet og størrelse
    circles = []
    img_filtered = img.copy()
    
    # Time hele filtreringsløkken
    start_filter = time.time()
    for i, contour in enumerate(contours):
        area = time_operation(filename, "detect_scoring_zones_tree", "cv2.contourArea", 
                             cv2.contourArea, contour)
        if area < config['min_contour_area']:  # Ignorer små contours
            continue
        
        # Sjekk hvor sirkel-lignende contour er
        perimeter = time_operation(filename, "detect_scoring_zones_tree", "cv2.arcLength", 
                                  cv2.arcLength, contour, True)
        if perimeter == 0:
            continue
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if circularity > config['circularity_threshold']:  # Terskel for sirkel-likhet
            (x, y), radius = time_operation(filename, "detect_scoring_zones_tree", "cv2.minEnclosingCircle", 
                                            cv2.minEnclosingCircle, contour)
            circles.append((int(x), int(y), int(radius)))
            # Tegn contour for visualisering
            time_operation(filename, "detect_scoring_zones_tree", "cv2.drawContours", 
                          cv2.drawContours, img_filtered, [contour], -1, config['color_green'], 
                          config['contour_thickness'])
    log_operation_time(filename, "detect_scoring_zones_tree", "contour_filtering_loop", time.time() - start_filter)
    
    save_visualization(img_filtered, "05_Filtrerte_contours", 5)
    
    # Sorter etter radius (største først = ytterste sone)
    start_sort = time.time()
    circles = sorted(circles, key=lambda x: x[2], reverse=True)
    log_operation_time(filename, "detect_scoring_zones_tree", "sorted(circles)", time.time() - start_sort)
    
    # Finn sentrum (gjennomsnitt av alle sentre, eller bruk den største)
    center = None
    if circles:
        start_mean = time.time()
        centers = np.array([(x, y) for x, y, r in circles])
        center = np.mean(centers, axis=0).astype(int)
        log_operation_time(filename, "detect_scoring_zones_tree", "np.mean(centers)", time.time() - start_mean)
    
    # Visualiser resultatet med nummererte sirkler
    result = img.copy()
    if circles:
        start_draw = time.time()
        for i, (x, y, r) in enumerate(circles):
            # Tegn sirkelen
            time_operation(filename, "detect_scoring_zones_tree", "cv2.circle", 
                          cv2.circle, result, (x, y), r, config['color_green'], 
                          config['circle_thickness'])
            # Tegn sentrum
            time_operation(filename, "detect_scoring_zones_tree", "cv2.circle", 
                          cv2.circle, result, (x, y), 2, config['color_red'], 3)
            # Legg til nummer (ytterste = 1, innerste = høyest)
            time_operation(filename, "detect_scoring_zones_tree", "cv2.putText", 
                          cv2.putText, result, str(i+1), (x-10, y), 
                          config['font'], config['font_scale_small'], config['color_blue'], 
                          config['font_thickness'])
        
        # Tegn hoved-sentrum
        if center is not None:
            time_operation(filename, "detect_scoring_zones_tree", "cv2.circle", 
                          cv2.circle, result, tuple(center), config['circle_center_size'], 
                          config['color_magenta'], config['circle_center_size'])
        log_operation_time(filename, "detect_scoring_zones_tree", "draw_result_loop", time.time() - start_draw)
    
    save_visualization(result, "06_Resultat_detect_scoring_zones_tree", 6)
    
    execution_time = time.time() - start_time
    log_performance(filename, "detect_scoring_zones_tree", "RETR_TREE", execution_time)
    
    # Skriv alle logg-oppføringer til fil
    flush_log_buffer()
    
    if visualize:
        plt.figure(figsize=config['figure_size'])
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        plt.imshow(result_rgb)
        plt.title(f'Detekterte poengsoner med RETR_TREE ({len(circles)} sirkler)')
        plt.axis('off')
        plt.show()
    
    return circles, center


if __name__ == "__main__":
    # Test med bilde fra config
    image_path = config['test_image_path']
    
    print("Tester find_outer_contour (RETR_EXTERNAL)...")
    circle, center = find_outer_contour(image_path, visualize=True)
    if circle:
        print(f"Funnet ytterste sirkel: sentrum=({circle[0]}, {circle[1]}), radius={circle[2]}")
    else:
        print("Ingen sirkel funnet")
    
    print("\nTester detect_scoring_zones_tree (RETR_TREE)...")
    circles, center = detect_scoring_zones_tree(image_path, visualize=True)
    print(f"Funnet {len(circles)} sirkler")
    if center is not None:
        print(f"Sentrum: ({center[0]}, {center[1]})")
