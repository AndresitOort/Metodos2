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

def sublists(list, sub_size = 10):
    sublist = []
    for i in range(0, len(list), sub_size):
        sublist.append(list[i:i + sub_size])
    return sublist

def delete_wdata(list,compl_list):
    listsb = sublists(list)
    compl_listsb = sublists(compl_list)
    
    for i in range(len(listsb)):
        psl = 1.8*prom_in_list(listsb[i])
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
        
    print(len(mergedlist),len(mergedcompl))
    
    return mergedlist, mergedcompl
    
IntensityN, WavelenghtN = delete_wdata(Intensity,Wavelenght)




plt.scatter(WavelenghtN,IntensityN,color='r',label='Scatter')
plt.scatter(Wavelenght,Intensity,color='b',label='Scatter',marker='x')

plt.plot(WavelenghtN,IntensityN,color='m',label='Scatter')
plt.legend()
plt.grid()
plt.show()

diferencias=[]
for i in range(1,len(Intensity)):
    diferencias.append(np.abs(Intensity[i]-Intensity[i-1]))

Wavelenght.pop(0)

promdel = prom_in_list(Intensity)

plt.scatter(Wavelenght,diferencias)
plt.axhline(promdel,color='r')
plt.grid()
plt.show()
