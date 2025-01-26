import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
import math

archivo = 'Rhodium.csv'
Wavelenght = pd.read_csv(archivo)['Wavelength (pm)'].tolist()
Intensity = pd.read_csv(archivo)['Intensity (mJy)'].tolist()

def prom_in_list(list):
    promdel = 0

    for i in range(len(list)-1):
        delt = np.abs(list[i+1] - list[i])
        promdel += delt
    
    promdel /= len(list)-1 

    return promdel

def sublists(list, sub_size = 6):
    sublist = []
    for i in range(0, len(list), sub_size):
        sublist.append(list[i:i + sub_size])
    return sublist

def delete_wdata(list,compl_list):
    listsb = sublists(list)
    compl_listsb = sublists(compl_list)
    
    for i in range(len(listsb)):
        psl = 1.896*prom_in_list(listsb[i])
        newdat = []
        newcompl = []
        for j in range(1,len(listsb[i])):
            if abs(listsb[i][j] - listsb[i][j-1]) <= psl:
                newdat.append(listsb[i][j-1])
                newcompl.append(compl_listsb[i][j-1])
            if j == len(listsb[i])-1 and abs(listsb[i][j] - listsb[i][j-1]) <= psl:
                newdat.append(listsb[i][j])
                newcompl.append(compl_listsb[i][j])
        listsb[i]=newdat
        compl_listsb[i] = newcompl
    mergedlist = []
    mergedcompl = []
    for sub in listsb:
        mergedlist.extend(sub)
    for sub in compl_listsb:
        mergedcompl.extend(sub)
        
    #print(len(mergedlist),len(mergedcompl))
    
    return np.array(mergedlist), np.array(mergedcompl)

# Punto 1.b)

def polinomio_lagrange(X_s,Y_s,x): # Creamos una función que nos admita una variable que usaremos como un symbol,
    # y un conjunto soporte definido por (X_s,Y_s) que van a ser los puntos por los cuales pase el polinomio que interpolemos.
    # Debemos crear una variable que almacene la suma de todos los polinomios de Lagrange.
    # Creamos también una variable que almacene el polinomio de Lagrange que, recordemos, es una productoria.
    
    
    polinomio_sum = np.float64(0)
    polinomio_Lagrange = np.float64(1)
    
    for n in range(len(X_s)): # El índice n lo usamos para calcular el Polinomio L_n de la base de lagrange.
        for i in range(len(X_s)): # El índice i lo usamos para recorrer cada elemento del conjunto soporte X_s.
            if i != n: # Para la base L_n del polinomio de Lagrange, preguntamos si X_s[i] != X_s[n]
                polinomio_Lagrange *= (x-X_s[i])/(X_s[n]-X_s[i]) # Calculamos la base de Lagrange con la Productoria al rededor de X_s[n]
        polinomio_sum += Y_s[n]*polinomio_Lagrange # Almacenamos la sumatoria de cada uno de los Polinomios de Lagrange
        polinomio_Lagrange = 1 # Reiniciamos la variable para calcular una nueva base.
    
    polinomio_sum = sym.simplify(polinomio_sum)
    
    return polinomio_sum

IntensityN, WavelenghtN = delete_wdata(Intensity,Wavelenght)

def derivative(domain,codomine):
    der_value = []
    
    for i in range(len(domain)-5):
        m = (codomine[i+5]-codomine[i])/(domain[i+5]-domain[i])
        der_value.append(m)
    
    return der_value
        
#plt.scatter(WavelenghtN,IntensityN,color='m')
#plt.show()


der = derivative(WavelenghtN,IntensityN)
newder = []
newint = []
WavelenghtNN = []
for i in range(len(der)):
    if abs(der[i]) <= 0.008 and IntensityN[i] <= 0.12:
        newder.append(der[i])
        newint.append(IntensityN[i])
        WavelenghtNN.append(WavelenghtN[i])
        

plt.plot(WavelenghtNN,newder,color='b')
plt.scatter(WavelenghtNN,newint,color='r')
#plt.scatter(WavelenghtN[:-5],newint,color='r',label='derivada')
#plt.plot(WavelenghtN,IntensityN,color='m',label='Modelo')
plt.legend()
plt.grid()
plt.show()

'''plt.scatter(WavelenghtN,IntensityN,color='r',label='Scatter')
plt.scatter(Wavelenght,Intensity,color='b',label='Original')

plt.plot(WavelenghtN,IntensityN,color='m',label='Scatter')
plt.legend()
plt.grid()
plt.show()'''
'''
diferencias=[]
for i in range(1,len(Intensity)):
    diferencias.append(np.abs(Intensity[i]-Intensity[i-1]))

Wavelenght.pop(0)

promdel = prom_in_list(Intensity)

plt.scatter(Wavelenght,diferencias)
plt.axhline(promdel,color='r')
plt.grid()
plt.show()
'''