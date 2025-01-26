import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi
from scipy.signal import savgol_filter, find_peaks
from Punto_1 import 

def punto1b(x, y):
    window_length = 51  
    polyorder = 3
    background_fit = savgol_filter(y, window_length, polyorder)

    isolated_peaks = y - background_fit

    peaks_indices, properties = find_peaks(isolated_peaks, height=0)
    x_peaks = x[peaks_indices]
    y_peaks = isolated_peaks[peaks_indices]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Espectro Original', color='blue', alpha=0.5)
    plt.plot(x, background_fit, label='Fondo Suavizado', color='orange', linestyle='--')
    plt.plot(x, isolated_peaks, label='Espectro sin Fondo', color='green', alpha=0.7)
    plt.scatter(x_peaks, y_peaks, color='red', label='Picos', zorder=5)
    plt.xlabel('Wavelength (pm)')
    plt.ylabel('Intensity')
    plt.title('Aislamiento de los Picos de Rayos X (Filtro Savitzky-Golay)')
    plt.legend()
    plt.grid(True)
    plt.show()