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
frec_real = frecuencias[picos]

#Ploteamos los módulos de cada punto:
modulos = np.mod(frec_real*t, 1)

#Identificamos lo especial de esta frecuencias. Si bien para cada dato hay un pequeño desfase, a diferencia de una frecuencia
#que no es representativa, aqui si se empaquetan en una misma figura,.
plt.scatter(modulos, h, s=5, color="green")
plt.title("2.a")
plt.xlabel(r"Fase Transformada General $\phi$")
plt.ylabel("Campo Magnético H")
plt.grid(True)
plt.show()
