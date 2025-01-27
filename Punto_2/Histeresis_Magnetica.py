import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
import math

from scipy.optimize import curve_fit

ruta = 'Punto_2\hysteresis.dat'
def leer_datos(ruta)->tuple:
    archivo = open(ruta,'r')
    t = []
    B = []
    H = []
    for line in archivo:
        line_0 = line.replace('\n',' ')
        line_1 = list(line_0)
        Data = []
        while len(line_1)>1:
            j = 0
            while line_1[j+1] != ' ' and line_1[j+1] != "-" and j <= 5:
                j+=1
            Data.append(float(''.join(line_1[:j+1])))
            line_1 = line_1[j+1:]
            
        t.append(Data[0])
        B.append(Data[1])
        H.append(Data[2])

    return (t,B,H)

def polinomio_lagrange(x,X_s,Y_s): # Creamos una función que nos admita una variable que usaremos como un symbol,
    # y un conjunto soporte definido por (X_s,Y_s) que van a ser los puntos por los cuales pase el polinomio que interpolemos.
    # Debemos crear una variable que almacene la suma de todos los polinomios de Lagrange.
    # Creamos también una variable que almacene el polinomio de Lagrange que, recordemos, es una productoria.
    
    polinomio_sum = 0
    polinomio_Lagrange = 1
    
    for n in range(len(X_s)): # El índice n lo usamos para calcular el Polinomio L_n de la base de lagrange.
        for i in range(len(X_s)): # El índice i lo usamos para recorrer cada elemento del conjunto soporte X_s.
            if i != n: # Para la base L_n del polinomio de Lagrange, preguntamos si X_s[i] != X_s[n]
                polinomio_Lagrange *= (x-X_s[i])/(X_s[n]-X_s[i]) # Calculamos la base de Lagrange con la Productoria al rededor de X_s[n]
        polinomio_sum += Y_s[n]*polinomio_Lagrange # Almacenamos la sumatoria de cada uno de los Polinomios de Lagrange
        polinomio_Lagrange = 1 # Reiniciamos la variable para calcular una nueva base.
        
    return polinomio_sum

def Interpolate(x,X,Y):
    
    Poly = 0
    
    for i in range(X.shape[0]):
        Poly += polinomio_lagrange(x,X,i)*Y[i]
        
    return Poly

a = leer_datos(ruta)

'''for i in a[0]:
    print(type(i))
'''
def sinusoidalfit(x,A,B):
    y = A*np.sin(B*x) #+ C*np.cos(B*x)
    return y

guess = [2.46,2.97]

param, cov = curve_fit(sinusoidalfit,a[0],a[1],p0 = guess)

param = param.tolist()

fit_sinu = []

print(param[1]/(2*np.pi))

for i in a[0]:
    fit_sinu.append(sinusoidalfit(i,param[0],param[1]))

def areaentrecurvas(B,H):
    sum = 0
    for i in range(len(B)-1):
        sum += ((B[i]+B[i+1])*(H[i+1]-H[i]))/2

    return sum

area = areaentrecurvas(a[1],a[2])

print(area)

'''plt.scatter(a[0],a[1],label='B vs t' , color='r')
plt.plot(a[0],fit_sinu)
plt.grid()
plt.legend()
plt.show()'''
'''plt.scatter(a[0],a[2],label='H vs t' , color='m')
plt.grid()
plt.legend()
plt.show()'''
plt.scatter(a[1],a[2],label='H vs B' , color='b')
plt.grid()
plt.legend()
plt.show()
