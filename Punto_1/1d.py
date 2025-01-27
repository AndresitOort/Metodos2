import csv
import numpy as np
import matplotlib.pyplot as plt
import Punto1Filtrado as p1f
import sympy as sym

def polinomio_lagrange(x,X_s,Y_s): # Creamos una función que nos admita una variable que usaremos como un symbol,
    # y un conjunto soporte definido por (X_s,Y_s) que van a ser los puntos por los cuales pase el polinomio que interpolemos.
    # Debemos crear una variable que almacene la suma de todos los polinomios de Lagrange.
    # Creamos también una variable que almacene el polinomio de Lagrange que, recordemos, es una productoria.
    
    polinomio_sum = 0
    polinomio_Lagrange = 1
    
    for n in range(len(X_s)): # Vamos a reccorer cada uno de los índices del conjunto soporte para calcular el polinomio 1 por 1.
        for i in range(len(X_s)):
            if i != n:
                polinomio_Lagrange *= (x-X_s[i])/(X_s[n]-X_s[i])
        polinomio_sum += Y_s[n]*polinomio_Lagrange
        polinomio_Lagrange = 1
    
    Lagrange = sym.sympify(polinomio_sum)
    print(Lagrange)
    Lagrange = sym.lambdify([x],Lagrange,'numpy')
    return polinomio_sum 

def Energia_total_Radiada(espectro_fondo,ventanas = 10):
    
    intervalo = int(300/ventanas)
    Roots, Weights = np.polynomial.legendre.leggauss(5)
    area_total = 0
    for i in range(ventanas):
        inf = i*intervalo
        sup = (i+1)*intervalo
        t = 0.5*( (sup-inf)*Roots + sup + inf)
        Integral = 0.5*(sup-inf)*np.sum(Weights*espectro_fondo(t))
        area_total += Integral    
    
    return area_total