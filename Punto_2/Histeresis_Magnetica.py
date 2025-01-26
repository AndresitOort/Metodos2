import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
import math

ruta = 'hysteresis.dat'
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
            while line_1[j+1] != ' ' and line_1[j+1] != "-":
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



    
