import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter, find_peaks


#----------------------------------------------------------- Punto 2.a
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

#Encontramos los módulos de cada punto:
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
        #Se pasa a fechas
df_2b["Date"] = pd.to_datetime(df_2b[["Year", "Month", "Day"]])

        #Escogemos hasta el 2010-01-01
df_2b_filtrado = df_2b[df_2b["Date"] <= pd.to_datetime("2010-01-01")]

manchas = df_2b_filtrado["SSN"].to_numpy()

#Eliminamos el DC
media = np.mean(manchas)
manchas = manchas - media

t_2b = np.arange(0, len(manchas),1)
fourier_2b = np.fft.rfft(manchas)
frecuencias_2b = np.fft.rfftfreq(len(manchas)) #[frec] = 1/dia
amplitud_2b = np.abs(fourier_2b)

#Encontramos la frecuencia caracteristica
index_picos_2b, info_2b = find_peaks(amplitud_2b, height=40000, distance=60) 
frec_real_2b = frecuencias_2b[index_picos_2b[0]]
y_pico_2b = info_2b["peak_heights"][0]

mods = np.mod(frec_real_2b*t_2b, 1)
T = 1/frec_real_2b
plt.scatter(mods, manchas, s=0.5)
plt.xlabel(r"Fase $\phi$")
plt.ylabel(r"Número Manchas")
plt.title(f"Empaquetado con T = {round(T)} dias")
plt.show()

print(f'2.b.a) P_solar = {T} dias')

#----------------------------------------------------------------------------- Punto 2.b.b

#Implementamos la transformada Inversa:

#Expandimos nuestro valor de tiempo
dif_t = (pd.to_datetime("2025-02-16") - pd.to_datetime("2010-01-01")).days

numero_armonicos = 50
t_2 = np.arange(0, len(manchas)+dif_t)
f = frecuencias_2b[:numero_armonicos]
X = fourier_2b[:numero_armonicos]
N = len(manchas)

def Transformada_Inversa(X, f, N, t_2):
    Y_t = []
    for t in t_2:
        suma = np.sum((X/2)*np.exp(2j*np.pi*f*t))
        y_t = (np.real(suma)/N) +media  #Importante sumarle el DC de nuevo
        Y_t.append(y_t)
    return Y_t

predicciones = Transformada_Inversa(X,f, N, t_2)

print(f'2.b.b) n_manchas_hoy = {predicciones[-1]}')

#Agregamos nuestra Prediccion a los datos en bruto

df_final = pd.DataFrame({"Date":pd.date_range(start=df_2b_filtrado["Date"].iloc[0], end=pd.to_datetime("2025-02-16") , freq='D'),
                         "SSN": predicciones})

#Graficamos (Nombre de la gráfica 2.b.b.pdf)

#fig, ax = plt.subplots(figsize=(10, 5))
#df_final.plot(x="Date", y="SSN", color="black", ax=ax)
#df_2b_filtrado.plot(x="Date", y="SSN", kind="scatter", ax=ax, s=0.5)
#plt.title(f"Reconstrucción de Señal con {numero_armonicos} armónicos")
#plt.show()