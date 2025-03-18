import pandas as pd
import numpy as np
import re
import unicodedata
import os
import random
import matplotlib.pyplot as plt

from numba import njit

s = "GTCTTAAAAGGCGCGGGTAAGGCCTTGTTCAACACTTGTCCCGTA"

a = list("ACGT")
c = list("ACGT")
F = pd.DataFrame(np.zeros((4,4),dtype=int),
 index=a,columns=c)
for i in range(len(s)-1):
 F.loc[s[i],s[i+1]] += 1
 
P = F.div(F.sum(axis=1).replace(0, 1), axis=0)
 
P = P.fillna(0)
 
nueva_letra = np.random.choice(a,p=P.loc[s[-1]].values)

print(nueva_letra)

ruta_guardado = r"Tarea_4"
nombre_libro = "The Shadow Over Innsmouth.txt"
nombre_limpio = "The Shadow Over Innsmouth cleaned.txt"

ruta_libro_original = os.path.join(ruta_guardado, nombre_libro)
ruta_libro_limpio = os.path.join(ruta_guardado, nombre_limpio)

with open("Tarea_4\\words_alpha.txt", "r", encoding="utf-8") as f:
    palabras_ingles = set(f.read().split())

def limpiar_texto(texto):
    texto = texto.replace("\r\n", "\n").replace("\n\n", "#").replace("\n", " ").replace("#", "\n\n")

    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("utf-8")

    texto = re.sub(r"[^a-zA-Z0-9\s\n]", "", texto)

    texto = texto.lower()

    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()

def procesar_libro(ruta_entrada, ruta_salida):
    with open(ruta_entrada, "r", encoding="utf-8") as f:
        texto = f.read()

    texto_limpio = limpiar_texto(texto)

    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write(texto_limpio)

procesar_libro(ruta_libro_original, ruta_libro_limpio)

def construir_tabla_frecuencias(texto, n=3):

    frecuencias = {}

    for i in range(len(texto) - n):
        n_grama = texto[i:i+n]  
        siguiente = texto[i+n]

        if n_grama not in frecuencias:
            frecuencias[n_grama] = {}

        if siguiente not in frecuencias[n_grama]:
            frecuencias[n_grama][siguiente] = 0

        frecuencias[n_grama][siguiente] += 1  

    df = pd.DataFrame(frecuencias).T.fillna(0)
    df = df.div(df.sum(axis=1), axis=0)  

    return df

def generar_texto(tabla, m=1500, n=3):
    texto_generado = ""

    posibles_inicios = [ngr for ngr in tabla.index if ngr.startswith("\n")]
    if posibles_inicios:
        n_grama = random.choice(posibles_inicios)
    else:
        n_grama = random.choice(tabla.index)

    texto_generado += n_grama

    for _ in range(m - n):
        if n_grama not in tabla.index: 
            break

        siguiente = np.random.choice(tabla.columns, p=tabla.loc[n_grama].fillna(0).values)
        texto_generado += siguiente

        n_grama = texto_generado[-n:]

    return texto_generado

def calcular_porcentaje_palabras(texto):
    palabras = texto.lower().split()
    palabras_validas = [p for p in palabras if p in palabras_ingles]
    return (len(palabras_validas) / len(palabras)) * 100 if palabras else 0

with open(ruta_libro_limpio, "r", encoding="utf-8") as f:
    texto = f.read()

resultados = {}

for n in range(1, 26): 
    print(f"Procesando n={n}...")
    tabla_frecuencias = construir_tabla_frecuencias(texto, n)
    texto_predicho = generar_texto(tabla_frecuencias, m=100000, n=n)

    ruta_salida = os.path.join(ruta_guardado, f"gen_text_n{n}.txt")
    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write(texto_predicho)

    porcentaje = calcular_porcentaje_palabras(texto_predicho)
    resultados[n] = porcentaje
    print(f"n = {n} → {porcentaje:.2f}% de palabras válidas en inglés")

plt.figure(figsize=(8,5))
plt.plot(resultados.keys(), resultados.values(), marker="o", linestyle="-", color="b")
plt.xlabel("Tamaño de n-grama (n)")
plt.ylabel("Porcentaje de palabras en inglés (%)")
plt.title("Influencia del tamaño del n-grama en la coherencia del texto")
plt.grid(True)
plt.xticks(range(1,26))

ruta_grafico = os.path.join(ruta_guardado, "4.pdf")
plt.savefig(ruta_grafico)
plt.show()

print(f"Gráfico guardado en: {ruta_grafico}")