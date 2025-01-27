import numpy as np
from Punto1bModelamiento import fondo, picos, max_x, max_y, x
import matplotlib.pyplot as plt




def FWHM(ymed, x, y):
    i=0
    x_med=[]
    while i < (len(x)-1):
        if y[i] == ymed:
            x_med.append(x[i])
        if (y[i+1]>ymed and y[i]<ymed) or (y[i]>ymed and y[i+1]<ymed):
            if abs(y[i+1]-ymed)  > abs(y[i]-ymed):
                x_med.append(x[i])
            else:
                x_med.append(x[i+1])
        i+=1
    
    fwhm= x_med[1]-x_med[0]
    return fwhm, x_med[0]

def Empaquetado(lista_a_ubicar, x, y): #Me devuelve las posiciones de los maximos
    y_res= []
    x_res= []
    for b in lista_a_ubicar:
        i=0
        while i < len(x):
            if y[i] == b:
                y_res.append(b)
                x_res.append(x[i])
                i=len(x)
            i+=1
    return x_res, y_res

maximo = max(fondo)
fwhm_fondo, lin_fondo = FWHM(maximo/2, x, fondo)
x_max_fondo, y_max_fondo = Empaquetado([maximo],x, fondo)

fwhm_pico_1, lin_pico_1 = FWHM(max_y[0]/2, x, picos)
fwhm_pico_2, lin_pico_2 = FWHM(max_y[1]/2, x, picos)

print(lin_pico_1)

def graficar_datos(x, y, x_resaltados, y_resaltados, x_linea_inicio, x_linea_fin, y_linea_valor):
    """
    Función para graficar datos con puntos resaltados y una línea recta entre dos puntos.
    
    Parámetros:
    - x: Conjunto de datos en el eje X (lista o array).
    - y: Conjunto de datos en el eje Y (lista o array).
    - x_resaltados: Valores del eje X a resaltar (lista o array).
    - y_resaltados: Valores del eje Y correspondientes a los puntos a resaltar (lista o array).
    - x_linea_inicio: Inicio del segmento en el eje X.
    - x_linea_fin: Fin del segmento en el eje X.
    - y_linea_valor: Valor constante del eje Y para la línea.
    """
    # Crear la gráfica
    plt.figure(figsize=(10, 6))

    # Graficar los datos originales
    plt.plot(x, y, label="Datos", color="blue")

    # Resaltar los puntos específicos
    plt.scatter(x_resaltados, y_resaltados, color="red", label="Datos resaltados", zorder=5)

    # Dibujar la línea recta entre dos puntos en el eje X
    plt.plot([x_linea_inicio, x_linea_fin], [y_linea_valor, y_linea_valor], 
             color="green", linestyle="--", label="Línea recta")

    # Personalizar la gráfica
    plt.title("Gráfica con puntos resaltados y línea recta")
    plt.xlabel("Eje X")
    plt.ylabel("Eje Y")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()

graficar_datos(x, fondo, x_max_fondo, y_max_fondo, lin_fondo, lin_fondo + fwhm_fondo, maximo/2)
graficar_datos(x, picos, max_x[0], max_y[0], lin_pico_1, lin_pico_1 + fwhm_pico_1, max_y[0]/2)




