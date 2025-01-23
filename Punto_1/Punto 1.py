import csv
import numpy as np
import matplotlib.pyplot as plt

# Ruta al archivo CSV
csv_file = 'Punto 1/Datos/Rhodium.csv'

# Listas para almacenar los datos
x_data = []
y_data = []

# Leer el archivo CSV
with open(csv_file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(spamreader)  # Saltar el encabezado
    for row in spamreader:
        x_data.append(float(row[0]))
        y_data.append(float(row[1]))

# Convertir a arrays de numpy
x = np.array(x_data)
y = np.array(y_data)



# Función para calcular barras de error y limpiar datos corruptos
def calcular_barras_de_error(x, y, ventana=5, umbral=2):
    y_limpio = []
    x_limpio = []

    for i in range(len(y)):
        if i < ventana or i > len(y) - ventana - 1:
            y_limpio.append(y[i])
            x_limpio.append(x[i])
        else:
            vecindad = y[i-ventana:i+ventana+1]
            promedio_local = np.mean(vecindad)
            desviacion_std = np.std(vecindad)


            if abs(y[i] - promedio_local) <= umbral * desviacion_std:
                y_limpio.append(y[i])
                x_limpio.append(x[i])
                barras_error.append(desviacion_std)

    return np.array(x_limpio), np.array(y_limpio), np.array(barras_error)


x_limpio, y_limpio, barras_error = calcular_barras_de_error(x, y, ventana=5, umbral=2)

plt.figure(figsize=(10, 6))
plt.errorbar(x_limpio, y_limpio, fmt='o', color='g', label='Datos Limpios')
plt.plot(x, y, marker='o', linestyle='-', color='b', alpha=0.5, label='Datos Originales')
plt.xlabel('Wavelength (pm)')
plt.ylabel('Intensity')
plt.title('Rhodium Data Plot - Barras de Error y Limpieza')
plt.legend()
plt.grid(True)
plt.show()
