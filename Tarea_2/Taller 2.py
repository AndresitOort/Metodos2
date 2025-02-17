import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


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
archivo = "OGLE-LMC-CEP-0001.dat"
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

print(f"1.c) f Nyquist: 0.35 MHz Hz")
print(f'1.c) f true: {frecuencia_real} Hz')