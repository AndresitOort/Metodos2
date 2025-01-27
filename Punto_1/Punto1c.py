import numpy as np

def FWHM(x,y):
    #Determinamos los máximos
    coordenada_máxima=[0,0]
    i=0
    while i< len(x):
        if y[i] >= coordenada_máxima[1]:
            coordenada_máxima[1] = y[i]
            coordenada_máxima[0] = x[i]
        i+=1
    
    #Encontramos el fmax/2
    ymed = coordenada_máxima[1]/2
    