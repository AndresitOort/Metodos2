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

def radiacion_fondo(x,C=29342324535.280624,B=-1365711523.2963758,T=-3865343.8688157196):
    return C * (x**(-5)) * (1 / (np.exp(B / (x * T)) - 1))
    
def Energia_total_Radiada(espectro_fondo,ventanas = 10):
    
    Roots, Weights = np.polynomial.legendre.leggauss(6)
    t = 0.5*( (300)*Roots+300)
    Integral = 0.5*(300)*np.sum(Weights*espectro_fondo(t))   
    
    return 'La energía total radiada corresponde a ({} +/- {}) eV/nm'.format(round(Integral,1),round(Integral*0.02,1))

x = np.linspace(0,300,500)
funcion_radiacion_fondo = radiacion_fondo(x,29342324535.280624,-1365711523.2963758,-3865343.8688157196)
energia = Energia_total_Radiada(radiacion_fondo)
print(energia)
'''plt.plot(x,funcion_radiacion_fondo,color ='m',label = 'Modelo Radiación de Fondo')
plt.grid()
plt.legend()
plt.show()'''