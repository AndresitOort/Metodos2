import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
import math

archivo = 'C:\\Users\\david\\OneDrive\\Documentos\\Universidad\\Programas\\Metodos2\\Metodos2\\Punto 1\\Datos\\Rhodium.csv'
Wavelenght = pd.read_csv(archivo)['Wavelength (pm)'].tolist()
Intensity = pd.read_csv(archivo)['Intensity (mJy)'].tolist()

promdel = 0

for i in range(len(Intensity)-1):
    delt = np.abs(Intensity[i+1] - Intensity[i])
    
    promdel += delt
    
promdel = promdel/len(Intensity)

print(promdel)

def delete_wdata(list,promd):
    newdata = []
    newwave = []
    
    prom = 1.8*promd
    
    for i in range(1,len(list)):
        if abs(list[i] - list[i-1]) <= prom:
            newdata.append(list[i-1])
            newwave.append(Wavelenght[i-1])
        elif abs(list[i] - list[i-1]) > prom:
            if abs(list[i+1]-list[i]) <= prom:
                newdata.append(list[i])
                newwave.append(Wavelenght[i])
            
    return newdata, newwave

IntensityN, WavelenghtN = delete_wdata(Intensity,promdel)

plt.scatter(WavelenghtN,IntensityN,color='r',label='Scatter')
plt.plot(WavelenghtN,IntensityN,color='m',label='Scatter')
plt.legend()
plt.grid()
plt.show()
