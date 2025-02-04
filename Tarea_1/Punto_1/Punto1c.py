import numpy as np
from Punto1bModelamiento import fondo, picos, max_y, x, indices
import matplotlib.pyplot as plt




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

#Sacamos la informacion del array fondo: Su máximo, sus coordenadas en la gráfica, y el indice de los arrays
maximo = max(fondo)
x_max_fondo, y_max_fondo, indice_max_fondo = Empaquetado([maximo], x, fondo)
fwhm_fondo, lin_fondo = FWHM(indice_max_fondo[0], x, fondo)

#Sacamos la informacion de los picos
x_max_picos, y_max_picos, indices_max_picos = Empaquetado(max_y, x, picos)
fwhm_pico_1, lin_pico_1 = FWHM(indices[0], x, picos)
fwhm_pico_2, lin_pico_2 = FWHM(indices[1], x, picos)

print(lin_pico_1)

def graficar_datos(x, y, x_resaltados, y_resaltados, x_linea_inicio, x_linea_fin, y_linea_valor, fwhm, titulo):
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
    plt.plot(x, y, label="Datos", color="blue", linestyle='--', linewidth= 0.6 )

    # Resaltar los puntos específicos
    plt.scatter(x_resaltados, y_resaltados, color="red", label="Datos resaltados", zorder=5)
    plt.text(x_resaltados + 1, y_resaltados, f"({x_resaltados}, {y_resaltados})", color="blue")

    # Dibujar la línea recta entre dos puntos en el eje X
    plt.plot([x_linea_inicio, x_linea_fin], [y_linea_valor, y_linea_valor], 
             color="green", linestyle="--", label="FWHM = {}".format(fwhm))

    # Personalizar la gráfica
    plt.title(titulo)
    plt.xlabel("Longitud de onda (λ) (pm)")
    plt.ylabel("Intensidad")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()

graficar_datos(x, fondo, x_max_fondo[0], y_max_fondo[0], lin_fondo, lin_fondo + fwhm_fondo, maximo/2, fwhm_fondo,"Gráfica aislada de la radiación de Fondo a partir de la aproximación por cuerpo negro. (Máximos y FWHM)" )
graficar_datos(x, picos, x_max_picos[1], y_max_picos[1], lin_pico_1, lin_pico_1 + fwhm_pico_1, max_y[1]/2, fwhm_pico_1, "Gráfica aislada del primer pico. (Máximos y FWHM)")
graficar_datos(x, picos, x_max_picos[0], y_max_picos[0], lin_pico_2, lin_pico_2 + fwhm_pico_2, max_y[0]/2, fwhm_pico_2, "Gráfica aislada del segundo pico. (Máximos y FWHM)")




