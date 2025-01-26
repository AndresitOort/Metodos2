import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

csv_file = 'Rhodium.csv'

x_data = []
y_data = []

with open(csv_file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(spamreader) 
    for row in spamreader:
        x_data.append(float(row[0]))
        y_data.append(float(row[1]))

x = np.array(x_data)
y = np.array(y_data)

def filtrar_por_ventanas(x, y, ventana=5, umbral=5):
    x_limpio = []
    y_limpio = []
    
    for i in range(len(y)):
        inicio = max(0, i - ventana)
        fin = min(len(y), i + ventana + 1)
        
        ventana_y = y[inicio:fin]
        
        mediana = np.median(ventana_y)
        mad = np.median(np.abs(ventana_y - mediana))
        if mad == 0:
            mad = 1e-6  
        
        robust_z = np.abs((y[i] - mediana) / (mad * 1.4826))
        
        if robust_z <= umbral:
            x_limpio.append(x[i])
            y_limpio.append(y[i])
    
    return np.array(x_limpio), np.array(y_limpio)

x_limpio, y_limpio = filtrar_por_ventanas(x, y, ventana=6, umbral=3.)

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


# Llamada corregida
#punto1b(x_limpio, y_limpio)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o-', color='b', alpha=0.5, label='Datos Originales')
plt.plot(x_limpio, y_limpio, 'o-', color='r', label='Datos Filtrados (Ventanas MAD)')
plt.xlabel('Wavelength (pm)')
plt.ylabel('Intensity')
plt.title('Rhodium Data Plot - Limpieza con Ventanas MAD')
plt.legend()
plt.grid(True)
plt.show()

print(x_limpio)

