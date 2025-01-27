import numpy as np
from Punto1bModelamiento import fondo, picos, x



def máximos(arr, inicio, dir_busqueda):
    i=inicio
    if inicio == -1:
        i=len(x)-1
    
    cord_max= [0,0] # coordenada i, max
    cache = []
    while (i<len(x)) and (i>=0) :
        if arr[i]> cord_max[1]:
            cord_max[1] = arr[i]
            cord_max[0] = i
            cache = []
        else:
            val=False
            cache.append(val)
        if len(cache)>=13:
            i=len(x)
        i=i+dir_busqueda
    return cord_max

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
    
    fwhm= x_med[1]-x_med[0]
    return fwhm

def max_searcher(x, y):

    data = []
    i=1
    while i<3:
        dir = 1
        maximo = máximos(y,dir,dir)
        fwhm = FWHM(maximo/2, x, y)
        data.append([maximo, fwhm])
        dir = dir*-1
        i+=1
    return data

print(max_searcher(x, picos))