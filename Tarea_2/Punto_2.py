import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter, find_peaks


#Lector de archivos
def csv_to_dict(filename):

    df = pd.read_csv(filename)  

    data_dict = {col: df[col].to_numpy() for col in df.columns }
    
    return data_dict

#Accedemos al archivo
data_dict = csv_to_dict("Tarea_2/H_field.csv")

t = data_dict["t"]
h = data_dict["H"]

#Graficamos los datos
#plt.scatter(x,y)
#plt.show()


#Le aplicamos fft
t_spaced = np.linspace(t[0], t[-1], len(t))

fourier = np.fft.rfft(h)
frecuencias = np.fft.rfftfreq(len(t_spaced), t_spaced[1]-t_spaced[0])

amplitud = np.abs(fourier)

#Cogemos la frecuencia más representativa (Se ve que es la unica mayor que 100):
picos, info = find_peaks(amplitud, height= 100)
print(frecuencias[picos])

plt.plot(frecuencias, amplitud)
plt.scatter(frecuencias[picos], info["peak_heights"], color="red")
plt.grid(True)
plt.show()
