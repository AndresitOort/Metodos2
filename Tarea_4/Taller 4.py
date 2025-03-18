import pandas as pd
import numpy as np
import re
import unicodedata
import os
import random
import matplotlib.pyplot as plt
from numba import njit
from matplotlib.animation import FFMpegWriter
import matplotlib.animation as animation




#-------------------------------------------------------------------Punto 2

# Logística
D1 = D2 = 50  
lambda_ = 670e-7  
A = 0.4  
a = 0.1  
d=0.1
N = 75000  # Número de muestras

# rango de z
z_vals = np.linspace(-0.4, 0.4, 100)  

# muestras random x,y
x_samples = np.random.uniform(-A/2, A/2, N)
y_samples = np.concatenate([
    np.random.uniform(-d/2, d/2, N//2),
    np.random.uniform(d/2, d/2 + a, N//2)
])

#fases de la exponencial compleja
phase_factor = (2 * np.pi / lambda_) * (D1 + D2)
quad_phase_x = (np.pi / (lambda_ * D1)) * x_samples[:, None]**2
quad_phase_y = (np.pi / (lambda_ * D1)) * y_samples[None, :]**2

# Calcular la integral de camino de Feynman con optimización
def feynman_intensity(z):
    screen_phase = (np.pi / (lambda_ * D2)) * (z - y_samples)**2
    phase = phase_factor + quad_phase_x - 2 * (np.pi / (lambda_ * D1)) * x_samples[:, None] * y_samples[None, :] + quad_phase_y + screen_phase
    integral = np.sum(np.exp(1j * phase), axis=1)
    return np.abs(np.sum(integral))**2 #magnitud del número complejo

I_feynman = np.array([feynman_intensity(z) for z in z_vals])
I_feynman /= I_feynman.max()  # Normalización

# Modelo clásico
theta = np.arctan(z_vals / D2)
I_classic = (np.cos(np.pi * d * np.sin(theta) / lambda_)**2) * (np.sinc(a * np.sin(theta) / lambda_)**2)
I_classic /= I_classic.max()  # Normalización

# Graficar
'''plt.figure(figsize=(8, 5))
plt.plot(z_vals, I_feynman, label='Feynman', linestyle='dashed')
plt.plot(z_vals, I_classic, label='Clásico')
plt.xlabel("Posición z (cm)")
plt.ylabel("Intensidad Normalizada")
plt.legend()
plt.title("Comparación de Intensidades - Modelo Clásico vs. Feynman")
plt.grid()
plt.show()'''




#------------------------------------------------------------------Punto 3
#Puntos Cambios realizados
N = 150  # Tamaño de la malla
J = 0.2  # Interacción entre espines
beta = 10  # Inverso de la temperatura
num_frames = 500  # Número de frames en la animación
iterations_per_frame = 400  # Iteraciones por frame
total_iterations = num_frames * iterations_per_frame  # Iteraciones totales

# Inicialización de la malla con valores aleatorios ±1
spins = np.random.choice([-1, 1], size=(N, N))

# Función para calcular el cambio de energía local ΔE
def delta_energy(spins, i, j):
    neighbors = (
        spins[(i+1) % N, j] + spins[i-1, j] +
        spins[i, (j+1) % N] + spins[i, j-1]
    )
    return 2 * J * spins[i, j] * neighbors


def metropolis_step(spins, beta):
    i, j = np.random.randint(0, N, size=2)  # Selecciona un espín aleatorio
    dE = delta_energy(spins, i, j)

    if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
        spins[i, j] *= -1  # Se acepta el cambio de espín


fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(spins, cmap='gray', animated=True)

def update(frame):
    for _ in range(iterations_per_frame):  # Realizar múltiples iteraciones por frame
        metropolis_step(spins, beta)
    im.set_array(spins)
    return [im]

#ani = animation.FuncAnimation(fig, update, frames=num_frames, blit=True)

# Guardar el video con FFmpeg
#video_filename = "ising_model.mp4"
#writer = FFMpegWriter(fps=30, bitrate=1800)
#ani.save(video_filename, writer=writer, dpi=300)
#plt.close(fig)

# Mostrar el video en Google Colab
#HTML(f"""
#<video width="50%" controls>
#  <source src="{video_filename}" type="video/mp4">
#</video>
#""")

# Descargar el video al equipo local
#from google.colab import files
#files.download(video_filename)

#------------------------------------------------------------------Punto 4

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

ruta_guardado = r"Tarea_4\\punto_4"
nombre_libro = "The Shadow Over Innsmouth.txt"
nombre_limpio = "The Shadow Over Innsmouth cleaned.txt"

ruta_libro_original = os.path.join(ruta_guardado, nombre_libro)
ruta_libro_limpio = os.path.join(ruta_guardado, nombre_limpio)

with open("Tarea_4\\punto_4\\words_alpha.txt", "r", encoding="utf-8") as f:
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
    texto_predicho = generar_texto(tabla_frecuencias, m=1000000, n=n)

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