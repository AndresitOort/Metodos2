import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter, find_peaks


#----------------------------------------------------------- Punto 2
#Leemos los archivos

df_2a = pd.read_csv("Tarea_2/H_field.csv")  
data_dict = {col: df_2a[col].to_numpy() for col in df_2a.columns}

#Accedemos al archivo

t = data_dict["t"]
h = data_dict["H"]


#Le aplicamos fft
t_spaced = np.linspace(t[0], t[-1], len(t))

fourier = np.fft.rfft(h)
frecuencias = np.fft.rfftfreq(len(t_spaced), t_spaced[1]-t_spaced[0])

amplitud = np.abs(fourier)

#Cogemos la frecuencia más representativa (Se ve que es la unica mayor que 100):
index_picos, info = find_peaks(amplitud, height= 100)
frec_real = frecuencias[index_picos]

#Ploteamos los módulos de cada punto:
modulos = np.mod(frec_real*t, 1)

#Ploteamos
        #Identificamos lo especial de esta frecuencias. Si bien para cada dato hay un pequeño desfase, a diferencia de una frecuencia
        #que no es representativa, aqui si se empaquetan en una misma figura.
#plt.scatter(modulos, h, s=5, color="green")
#plt.title("2.a")
#plt.xlabel(r"Fase Transformada General $\phi$")
#plt.ylabel("Campo Magnético H")
#plt.grid(True)
#plt.show()

#------------------------------------------------------------------ Punto 2b
#Leemos y limpiamos el archivo
        #El separador ahora son tabulaciones, por eso el "\s+". Como la primera fila no es columna, la skipeamos.
df_2b = pd.read_csv("Tarea_2\list_aavso-arssn_daily.txt", sep="\s+", skiprows=1)
df_2b["Date"] = pd.to_datetime(df_2b[["Year", "Month", "Day"]])
        #Escogemos hasta el 2010-01-01
df_2b_filtrado = df_2b[df_2b["Date"] <= pd.to_datetime("2010-01-01")]

#Graficamos los datos
df_2b_filtrado.plot(x="Date", y="SSN")
plt.show()
#Le aplicamos Fourier a las señales
manchas = df_2b_filtrado["SSN"].to_numpy()
print(manchas.shape)
fourier_2b = np.fft.rfft(manchas)
frecuencias_2b = np.fft.rfftfreq(len(manchas)) #[frec] = 1/dia
amplitud_2b = np.abs(fourier_2b)
t_2b = np.linspace(0, len(manchas), 23742)

#plt.plot(np.log(frecuencias_2b), np.log(amplitud_2b))
#plt.xlabel("Frecuencias log")
#plt.ylabel("Amplitud log")
#plt.show
#Encontramos la frecuencia caracteristica
index_picos_2b, info_2b = find_peaks(amplitud_2b, height=40000, distance=60) 
#Respuesta de los indices [  6 879]; cogemos el segundo pues el primero corresponde al DC. La altura del segundo es 41093.59300372

plt.plot(np.log(frecuencias_2b), np.log(amplitud_2b))
plt.scatter(np.log(frecuencias_2b[index_picos_2b[1]]),np.log(amplitud_2b[index_picos_2b[1]]) )
plt.xlabel("Frecuencias log")
plt.ylabel("Amplitud log")
plt.show

frec_real_2b = frecuencias_2b[index_picos_2b[1]]
modulos_2b = np.mod(frec_real_2b*t_2b,1)
plt.scatter(modulos_2b, manchas, s=0.5)
plt.show()




