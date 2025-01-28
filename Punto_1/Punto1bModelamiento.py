import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi
from scipy.signal import savgol_filter, find_peaks
from  Punto1aFiltrado import x_limpio,y_limpio
from scipy.optimize import curve_fit

x= x_limpio
y= y_limpio

# def punto1b(x, y):
#     window_length = 51  
#     polyorder = 3
#     background_fit = savgol_filter(y, window_length, polyorder)

#     isolated_peaks = y - background_fit

#     peaks_indices, properties = find_peaks(isolated_peaks, height=0)
#     x_peaks = x[peaks_indices]
#     y_peaks = isolated_peaks[peaks_indices]

#     plt.figure(figsize=(10, 6))
#     plt.plot(x, y, label='Espectro Original', color='blue', alpha=0.5)
#     plt.plot(x, background_fit, label='Fondo Suavizado', color='orange', linestyle='--')
#     plt.plot(x, isolated_peaks, label='Espectro sin Fondo', color='green', alpha=0.7)
#     plt.scatter(x_peaks, y_peaks, color='red', label='Picos', zorder=5)
#     plt.xlabel('Wavelength (pm)')
#     plt.ylabel('Intensity')
#     plt.title('Aislamiento de los Picos de Rayos X (Filtro Savitzky-Golay)')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
    

def modelo_combinado(x, C, B, T):
    return C * (x**(-5)) * (1 / (np.exp(B / (x * T)) - 1))


def procesar_espectro(x, y):

    popt, _ = curve_fit(modelo_combinado, x, y, p0=(1, 1, 300), maxfev=50000)
    fondo_combinado = modelo_combinado(x, *popt)
    isolated_peaks = y - fondo_combinado
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Espectro Original', color='blue', alpha=0.5)
    plt.plot(x, fondo_combinado, label='Fondo Ajustado (Ley de Cuerpo Negro)', color='orange', linestyle='--')
    plt.plot(x, isolated_peaks, label='Espectro sin Fondo', color='green', alpha=0.7)
    plt.xlabel('Longitud de onda (λ) (pm)')
    plt.ylabel('Intensidad')
    plt.title('Aislamiento de los Picos de Rayos X (Ajuste de Fondo con Ley de Cuerpo Negro)')
    plt.legend()
    plt.grid(True)
    plt.show()
    ecuacion = f"f(x) = {popt[0]:.4f} * (x^(-5)) * (1 / (exp({popt[1]:.4f} / (x * {popt[2]:.4f}))) - 1)"
    return fondo_combinado, isolated_peaks, popt, ecuacion

fondo, picos, parametros, ecuacion = procesar_espectro(x_limpio, y_limpio)

print(f'Parámetros ajustados: C={parametros[0]}, B={parametros[1]}, T={parametros[2]}')
print(f'Ecuación ajustada: {ecuacion}')

peaks_indices, properties= find_peaks(picos, height=0)
sorted_peaks = np.sort(properties['peak_heights'])
#Escoger los primeros 3
max = sorted_peaks[-3:]
max_y = max[::2]

indices = []
for i in peaks_indices:
    if (picos[i] == max_y[0]) or (picos[i] == max_y[1]):
        indices.append(i)
print(indices)
#print(max(picos))