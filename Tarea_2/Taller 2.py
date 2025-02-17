import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.typing import NDArray
from scipy.signal import savgol_filter, find_peaks

def datos_prueba(t_max:float, dt:float, amplitudes:NDArray[float],
  frecuencias:NDArray[float], ruido:float=0.0) -> NDArray[float]:
  ts = np.arange(0.,t_max,dt)
  ys = np.zeros_like(ts,dtype=float)
  for A,f in zip(amplitudes,frecuencias):
    ys += A*np.sin(2*np.pi*f*ts)
  ys += np.random.normal(loc=0,size=len(ys),scale=ruido) if ruido else 0
  return ts,ys

#1a---------------------------------------------------------------------------------

def Fourier(t:NDArray[float], y:NDArray[float], f:float) -> complex:
  FFT = np.sum(np.array([a*np.exp((-2*np.pi*f*p)*1j) for a,p in zip(y,t)]))
  return FFT

t_max=15
d_t=0.1
frecuencias=[0.25,0.5,0.75]
amplitudes=[1,2,3]

frecuencias_grafica=np.linspace(0,7.5,1000)

#señales generadas
ts_sin_ruido, ys_sin_ruido = datos_prueba(t_max, d_t, amplitudes, frecuencias, ruido=0)
ts_con_ruido, ys_con_ruido = datos_prueba(t_max, d_t, amplitudes, frecuencias, ruido=2)

#transformadas
transformada_sin_ruido = np.array([Fourier(ts_sin_ruido, ys_sin_ruido, f) for f in frecuencias_grafica])
transformada_con_ruido = np.array([Fourier(ts_con_ruido, ys_con_ruido, f) for f in frecuencias_grafica])


print('1.a) Aparecen frecuencias que no son de la señal original.')


#1b---------------------------------------------------------------------------------------------------

def FWHM(inicio, x, y):
    ymed = y[inicio]/2
    """
        Esta función busca hacia ambas direcciones a partir del indice de un máximo de un array,
        buscando un valor muy cercano a ymax/2.
        devuelve el fwhm, y el valor inicial del intervalo.
    """
    #Busqueda hacia la izquierda
    x_med=[]
    i = inicio
    while i > 1:
        if y[i] == ymed:
            x_med.append(x[i])
            i = 1
        if (y[i]>ymed and y[i-1]<ymed):
            if abs(y[i+1]-ymed)  > abs(y[i]-ymed):
                x_med.append(x[i])
            else:
                x_med.append(x[i+1])
            i=1
        i-=1
    #Buscamos hacia la derecha

    i = inicio
    while i < len(x):
        if y[i] == ymed:
            x_med.append(x[i])
            i=len(x)
        if (y[i]>ymed and y[i+1]<ymed):
            if abs(y[i+1]-ymed)  > abs(y[i]-ymed):
                x_med.append(x[i])
            else:
                x_med.append(x[i+1])
            i=len(x)
        i+=1
    fwhm = x_med[1] - x_med[0]
    return fwhm, x_med[0]




def Empaquetado(lista_a_ubicar, x, y): #Me devuelve las posiciones de los maximos
    y_res= []
    x_res= []
    indice = []
    for b in lista_a_ubicar:
        i=0
        while i < len(x):
            if y[i] == b:
                y_res.append(b)
                x_res.append(x[i])
                indice.append(i)
                i=len(x)
            i+=1
    return x_res, y_res, indice

#Se pone en comentarios para que no se ejecute el código por la demora del mismo

'''frecuencia_1b=[0.5]
amplitud_1b=[1]
t_max_1b=np.linspace(10,300,39)
lista_fwhm=[]
frecuencias_grafica_1b=np.linspace(0,5,100)
for i in t_max_1b:
  ts_1b, ys_1b = datos_prueba(i, d_t, amplitud_1b, frecuencia_1b, ruido=0)

  transformada_1b=abs(np.array([Fourier(ts_1b, ys_1b, f) for f in frecuencias_grafica_1b]))

  maximo=max(transformada_1b)
  x_pico, y_pico, indice_max_pico = Empaquetado([maximo], frecuencias_grafica_1b, transformada_1b)
  fwhm_pico, lin_pico = FWHM(indice_max_pico[0], frecuencias_grafica_1b, transformada_1b)
  lista_fwhm.append(fwhm_pico)'''

print('El ajuste de la gráfica 1.b fue: 0.181*exp(-0.0536t) + 0.00735')



#1c--------------------------------------------------------------------------

import numpy as np
from scipy.signal import find_peaks
import random
archivo = "Tarea_2\OGLE-LMC-CEP-0001.dat"
tiempo=[]
intensidad=[]
incertidumbre=[]
#Leo el archivo
with open(archivo, "r") as file:
    for linea in file:
        # Evitar líneas vacías o comentarios (si las hay)
        if linea.strip() and not linea.startswith("#"):  # Omite líneas vacías o que comienzan con '#'
            # Separar la línea por espacios (o puedes usar .split() si los valores están separados por espacios o tabuladores)
            datos = linea.split()
            if len(datos) >= 2:  # Asegurarse de que haya al menos dos columnas
                # Convertir las primeras dos columnas a tipo numérico
                tiempo.append(float(datos[0]))
                intensidad.append(float(datos[1]))
                incertidumbre.append(float(datos[1]))
                
                
promedio=np.mean(intensidad)
intensidad_=np.array([i-promedio for i in intensidad])
frecuencias_grafica_1c=np.linspace(0, 5,100)
transformada_1c=(np.array(np.abs([Fourier(tiempo,intensidad_,f) for f in frecuencias_grafica_1c])))

frecuencia_real=frecuencias_grafica_1c[np.argmax(transformada_1c)]
fase=np.array([(frecuencia_real*t)%(1) for t in tiempo])

print(f"1.c) f Nyquist:  Hz")
print(f'1.c) f true: {frecuencia_real}')
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
modulos_fft = np.mod(frec_real*t, 1)

#Le aplicamos ft
f_list = np.linspace(0,10,500)
fourier_implemented = np.array([Fourier(t, h, f) for f in f_list])
amplitud_implemented  = np.abs(fourier_implemented)

indice_frec_real = np.argmax(amplitud_implemented)
print(frec_real)
frec_real_implemented = f_list[indice_frec_real]
print(frec_real_implemented)
modulos_ft_implemented = np.mod(frec_real_implemented*t, 1)



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
#Graficamos

#plt.scatter(mods, manchas, s=0.5)
#plt.xlabel(r"Fase $\phi$")
#plt.ylabel(r"Número Manchas")
#plt.title(f"Empaquetado con T = {round(T)} dias")
#plt.show()

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

#------------------------------------------------------------------------PUNTO 3----------------------------------------------------------------------


#Cargar datos
manchas_sol = pd.read_csv(r"Tarea_2\list_aavso-arssn_daily.txt", sep=r"\s+", skiprows=1)
        #Se pasa a fechas
manchas_sol["Date"] = pd.to_datetime(manchas_sol[["Year", "Month", "Day"]])

        #Escogemos hasta el 2010-01-01
manchas_sol = manchas_sol[manchas_sol["Date"] <= pd.to_datetime("2010-01-01")]

manchas_sol = manchas_sol["SSN"].to_numpy()
fft_signal = np.fft.fft(manchas_sol) #transformada de furier rapida
freq = np.fft.fftfreq(len(fft_signal)) #Fracuencias para cada punto
#Defino varios alfas para el filtro gauseano
alphas = [1, 100, 5, 10000, 25000]

resultados_filtrado = []

# fig, axs = plt.subplots(len(alphas), 2, figsize=(12, 3 * len(alphas)))

for i, alpha in enumerate(alphas):
    # El filtro
    filtro = np.exp(- (freq * alpha) ** 2)

    # Aplico el filtro
    fft_filtrada = fft_signal * filtro
    senal_filtrada = np.fft.ifft(fft_filtrada)
    resultados_filtrado.append(fft_filtrada)

    # Graficar 
    # axs[i, 0].plot(manchas_sol, label='Señal Original', alpha=0.7)
    # axs[i, 0].plot(senal_filtrada.real, label='Señal Filtrada', linestyle='--')
    # axs[i, 0].set_title(f'Señal Original vs Filtrada (alpha={alpha})')
    # axs[i, 0].legend()

    # axs[i, 1].plot(freq, np.abs(fft_signal), label='FFT Original', alpha=0.7)
    # axs[i, 1].plot(freq, np.abs(fft_filtrada), label='FFT Filtrada', linestyle='--')
    # axs[i, 1].set_title(f'Transformada de Fourier (alpha={alpha})')
    # axs[i, 1].legend()

# plt.tight_layout()
# plt.show()

# print(resultados_filtrado)


#---------------------------------------------------------------------IMAGENES------------------------------------------------

#cargar imegen como array
catto = plt.imread(r"Tarea_2\catto.png")

# Se reliso la ransfiormada de furier y luego se cwentraron la frecuencias para asi identificiar patrones que sean repetitibos par asi
# eliminar este ruido. Para esta imagen ajustamos dos iagonales una abajo y otra arriba y tambien empleamos una elimpce para protejer datos de frecuencias
# fundamentales que no se deben borrar. Sereemplaza el valor de la as frecuencias encerrada por cero dado que son la que se quieren elimina.

def diagonal_filter_catto(shape, angle=-45, width=10, value=0, desplazamiento_sup=-120, desplazamiento_inf=-80, a=11, b=900):
    rows, cols = shape[:2]
    filtro = np.ones((rows, cols))

    # Diagonal superior
    for i in range(rows // 2):
        for j in range(cols):
            x = i - rows / 2
            y = j - cols / 2
            if abs(i - (j - rows / 2 + desplazamiento_sup) * np.tan(np.radians(angle))) < width and (x/a)**2 + (y/b)**2 > 1:
                filtro[i, j] = value  

    #diagonal inferior
    for i in range(rows // 2, rows):
        for j in range(cols):
            x = i - rows / 2
            y = j - cols / 2
            if abs(i - (j - rows / 2 + desplazamiento_inf) * np.tan(np.radians(angle))) < width and (x/a)**2 + (y/b)**2 > 1:
                filtro[i, j] = value 

    return filtro


#aplica el filtro
def aplicar_notch_filter_catto(img, D0, notch_centers, angle, width,diagonal=False):
    f_transformada = np.fft.fft2(img)
    fshift = np.fft.fftshift(f_transformada)
    filtro = diagonal_filter_catto(img.shape, angle, width, value=0) 
    fshift_filtrado = fshift * filtro
    f_ishift = np.fft.ifftshift(fshift_filtrado)
    img_filtrada = np.fft.ifft2(f_ishift).real

    return img_filtrada, fshift_filtrado, filtro

#Parametros de las coordenadas de las diagonales
notch_centers = [(256, 300), (256, 212)]  

D0 = 15   #anchura de las diagonales


img_filtrada, espectro_filtrado, diagonal_mask = aplicar_notch_filter_catto(catto, D0, notch_centers, diagonal=True, angle=-75, width=90)

# fig, axs = plt.subplots(1, 3, figsize=(20, 6))

# axs[0].imshow(catto, cmap='gray')
# axs[0].set_title('Imagen Original')
# axs[0].axis('off')

# axs[1].imshow(np.log(1 + np.abs(espectro_filtrado)), cmap='gray')
# axs[1].set_title('Espectro Filtrado')
# axs[1].axis('off')

# axs[2].imshow(img_filtrada, cmap='gray')
# axs[2].set_title('Imagen Filtrada')
# axs[2].axis('off')

# plt.tight_layout()
# plt.show()

#IMAGEN 2

casttle = plt.imread(r"Tarea_2\Noisy_Smithsonian_Castle.jpg")

def notch_filter_casttle(shape, pos_izq, pos_der, pos_sup, pos_inf, width, value=0, a=20, b=9000):
    rows, cols = shape[:2]
    filtro = np.ones((rows, cols))

    for i in range(rows):
        for j in range(cols):
            x = i - rows / 2
            y = j - cols / 2

            fuera_elipse = (x/a)**2 + (y/b)**2 > 1

            en_linea_izq = abs(j - pos_izq) < width // 2
            en_linea_der = abs(j - pos_der) < width // 2
            en_linea_sup = abs(i - pos_sup) < width // 2
            en_linea_inf = abs(i - pos_inf) < width // 2

            if (en_linea_izq or en_linea_der or en_linea_sup or en_linea_inf) and fuera_elipse:
                filtro[i, j] = value

    return filtro

def eliminar_cuadrados(filtro, centros, tamanio):
    for (x_centro, y_centro) in centros:
        x_min = max(0, x_centro - tamanio // 2)
        x_max = min(filtro.shape[0], x_centro + tamanio // 2)
        y_min = max(0, y_centro - tamanio // 2)
        y_max = min(filtro.shape[1], y_centro + tamanio // 2)

        filtro[x_min:x_max, y_min:y_max] = 0

    return filtro

def aplicar_notch_filter_casttle(img, pos_izq, pos_der, pos_sup, pos_inf, width, cuadrados):
    f_transformada = np.fft.fft2(img)
    fshift = np.fft.fftshift(f_transformada)
    filtro = notch_filter_casttle(img.shape, pos_izq=pos_izq, pos_der=pos_der, pos_sup=pos_sup, pos_inf=pos_inf, width=width)
    filtro = eliminar_cuadrados(filtro, cuadrados, tamanio=10)
    fshift_filtrado = fshift * filtro
    f_ishift = np.fft.ifftshift(fshift_filtrado)
    img_filtrada = np.fft.ifft2(f_ishift).real

    return img_filtrada, fshift_filtrado, filtro

pos_izq = 410
pos_der = 610
pos_sup = 320
pos_inf = 410
width = 78
cuadrados = [(380, 415), (380, 615)]  


img_filtrada, espectro_filtrado, filtro_mask = aplicar_notch_filter_casttle(casttle, pos_izq, pos_der, pos_sup, pos_inf, width, cuadrados)

# fig, axs = plt.subplots(1, 3, figsize=(20, 6))

# axs[0].imshow(casttle, cmap='gray')
# axs[0].set_title('Imagen Original')
# axs[0].axis('off')

# axs[1].imshow(np.log(1 + np.abs(espectro_filtrado)), cmap='gray')
# axs[1].set_title('Espectro Filtrado')
# axs[1].axis('off')

# axs[2].imshow(img_filtrada, cmap='gray')
# axs[2].set_title('Imagen Filtrada')
# axs[2].axis('off')

# plt.tight_layout()
# plt.show()