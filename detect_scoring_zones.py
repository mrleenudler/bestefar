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


# Global variabel for å holde styr på visualiseringsindeks
visualization_index = 0
visualization_dir = Path("Visualiseringer")


def save_visualization(image, title, index=None):
    """
    Lagrer en visualisering som bilde i Visualiseringer katalogen.
    
    Args:
        image: Bildet å lagre (BGR eller RGB)
        title: Beskrivende tittel
        index: Indeks for rekkefølge (hvis None, brukes global index)
    """
    global visualization_index
    
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
    
    # Konverter til RGB hvis nødvendig (matplotlib forventer RGB)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Sjekk om det er BGR (OpenCV) eller RGB
        # Vi antar BGR hvis det kommer fra OpenCV
        if isinstance(image, np.ndarray) and image.dtype == np.uint8:
            # Konverter BGR til RGB for lagring
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
    else:
        image_rgb = image
    
    # Lagre bildet
    if len(image_rgb.shape) == 2:  # Gråskala
        plt.imsave(filepath, image_rgb, cmap='gray')
    else:  # Farge
        plt.imsave(filepath, image_rgb)
    
    print(f"Lagret visualisering: {filepath}")
    return filepath


def log_performance(filename, method_name, algorithm, execution_time):
    """
    Logger ytelsesdata til to filer: kronologisk og alfabetisk.
    
    Args:
        filename: Navn på bildet som ble prosessert
        method_name: Navn på metoden/funksjonen
        algorithm: Navn på algoritmen
        execution_time: Kjøretid i sekunder
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{timestamp} | {filename} | {method_name} | {algorithm} | {execution_time:.4f}s\n"
    
    # Kronologisk fil (append)
    with open("ytelse_kronologisk.txt", "a", encoding="utf-8") as f:
        f.write(entry)
    
    # Alfabetisk fil (les, legg til, sorter, skriv)
    entries = []
    if os.path.exists("ytelse_alfabetisk.txt"):
        with open("ytelse_alfabetisk.txt", "r", encoding="utf-8") as f:
            entries = f.readlines()
    
    entries.append(entry)
    # Sorter alfabetisk (basert på hele linjen)
    entries.sort()
    
    with open("ytelse_alfabetisk.txt", "w", encoding="utf-8") as f:
        f.writelines(entries)


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
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Kunne ikke lese bildet: {image_path}")
    
    save_visualization(img, "01_Originalt_bilde")
    
    # Konverter til gråskala
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save_visualization(gray, "02_Graaskala", 2)
    
    # Threshold for å få binær bilde
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    save_visualization(thresh, "03_Threshold_binar", 3)
    
    # Finn contours med RETR_EXTERNAL (kun ytterste contours)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Visualiser alle contours
    img_contours = img.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    save_visualization(img_contours, "04_Alle_contours_RETR_EXTERNAL", 4)
    
    # Finn den største konturen
    if not contours:
        execution_time = time.time() - start_time
        log_performance(filename, "find_outer_contour", "RETR_EXTERNAL", execution_time)
        return None, None
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Visualiser største contour
    img_largest = img.copy()
    cv2.drawContours(img_largest, [largest_contour], -1, (0, 255, 0), 3)
    save_visualization(img_largest, "05_Storste_contour", 5)
    
    # Bruk minEnclosingCircle på den største konturen
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center = (int(x), int(y))
    radius = int(radius)
    
    # Visualiser resultatet
    result = img.copy()
    cv2.circle(result, center, radius, (0, 255, 0), 3)
    cv2.circle(result, center, 5, (0, 0, 255), -1)
    cv2.putText(result, f"Radius: {radius}", (center[0] - 50, center[1] - radius - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    save_visualization(result, "06_Resultat_find_outer_contour", 6)
    
    execution_time = time.time() - start_time
    log_performance(filename, "find_outer_contour", "RETR_EXTERNAL", execution_time)
    
    if visualize:
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
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
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Kunne ikke lese bildet: {image_path}")
    
    save_visualization(img, "01_Originalt_bilde")
    
    # Konverter til gråskala
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save_visualization(gray, "02_Graaskala", 2)
    
    # Threshold for å få binær bilde
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    save_visualization(thresh, "03_Threshold_binar", 3)
    
    # Finn contours med RETR_TREE (alle contours, inkludert nested)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Visualiser alle contours
    img_contours = img.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    save_visualization(img_contours, "04_Alle_contours_RETR_TREE", 4)
    
    # Filtrer contours basert på sirkel-likhet og størrelse
    circles = []
    img_filtered = img.copy()
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 100:  # Ignorer små contours
            continue
        
        # Sjekk hvor sirkel-lignende contour er
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if circularity > 0.5:  # Terskel for sirkel-likhet (lavere enn før for å få flere)
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circles.append((int(x), int(y), int(radius)))
            # Tegn contour for visualisering
            cv2.drawContours(img_filtered, [contour], -1, (0, 255, 0), 2)
    
    save_visualization(img_filtered, "05_Filtrerte_contours", 5)
    
    # Sorter etter radius (største først = ytterste sone)
    circles = sorted(circles, key=lambda x: x[2], reverse=True)
    
    # Finn sentrum (gjennomsnitt av alle sentre, eller bruk den største)
    center = None
    if circles:
        centers = np.array([(x, y) for x, y, r in circles])
        center = np.mean(centers, axis=0).astype(int)
    
    # Visualiser resultatet med nummererte sirkler
    result = img.copy()
    if circles:
        for i, (x, y, r) in enumerate(circles):
            # Tegn sirkelen
            cv2.circle(result, (x, y), r, (0, 255, 0), 2)
            # Tegn sentrum
            cv2.circle(result, (x, y), 2, (0, 0, 255), 3)
            # Legg til nummer (ytterste = 1, innerste = høyest)
            cv2.putText(result, str(i+1), (x-10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Tegn hoved-sentrum
        if center is not None:
            cv2.circle(result, tuple(center), 5, (255, 0, 255), 5)
    
    save_visualization(result, "06_Resultat_detect_scoring_zones_tree", 6)
    
    execution_time = time.time() - start_time
    log_performance(filename, "detect_scoring_zones_tree", "RETR_TREE", execution_time)
    
    if visualize:
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(f'Detekterte poengsoner med RETR_TREE ({len(circles)} sirkler)')
        plt.axis('off')
        plt.show()
    
    return circles, center


if __name__ == "__main__":
    # Test med "Blink 1 (fake).jpg"
    image_path = Path("Blink 1 (fake).jpg")
    
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
