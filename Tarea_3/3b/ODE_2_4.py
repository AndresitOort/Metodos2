import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import matplotlib.animation as animation

@njit
def fuerza_coulomb(r):
    r_norm = np.linalg.norm(r)
    return -r / r_norm**3

@njit
def derivadas(t, y):
    r = np.array([y[0], y[1]])
    v = np.array([y[2], y[3]])
    a = fuerza_coulomb(r)
    return np.array([v[0], v[1], a[0], a[1]])

@njit
def runge_kutta4(f, y0, t):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    dt = t[1] - t[0]
    for i in range(n - 1):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + dt/2, y[i] + dt*k1/2)
        k3 = f(t[i] + dt/2, y[i] + dt*k2/2)
        k4 = f(t[i] + dt, y[i] + dt*k3)
        y[i+1] = y[i] + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    return y

# Condiciones iniciales
y0 = np.array([1.0, 0.0, 0.0, 1.0])
T_teo = 2 * np.pi  # Tercera ley de Kepler en unidades atómicas 
                    #(Asumiendo masa reducida (masa electrón) y teniendo que nuestra fuerza centripeta es 1/R^2 por nuestras unidades
                    #(Como R^2 es a0^2, queda que F = 1) solo sobrevive el término 4*pi^2 que se va con la raíz del periodo)

# Tiempo de simulación
t_max = 10  # Debe cubrir al menos un periodo
n_pasos = 1000

t = np.linspace(0, t_max, n_pasos)
y = runge_kutta4(derivadas, y0, t)

# Extraer posiciones y velocidades
global x, y_pos
x, y_pos, vx, vy = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
print(x)
# Calcular el período simulado
cambios_signo = np.where(np.diff(np.sign(y_pos)))[0]
if len(cambios_signo) > 1:
    P_sim = 2 * (t[cambios_signo[1]] - t[cambios_signo[0]])
else:
    P_sim = np.nan  # Si no hay suficiente precisión, no se calcula

# Convertir a attosegundos (1 unidad atómica de tiempo ≈ 24.19 as)
P_teo_as = T_teo * 24.19
P_sim_as = P_sim * 24.19

# Imprimir resultado
print(f"2.a) {{P_teo = {P_teo_as:.5f}; P_sim = {P_sim_as:.5f}}}")

# Verificación de radio y energía constantes
radio = np.sqrt(x**2 + y_pos**2)
energia_cinetica = 0.5 * (vx**2 + vy**2)
energia_potencial = -1 / radio
energia_total = energia_cinetica + energia_potencial

# Graficar órbita
plt.figure(figsize=(6, 6))
plt.plot(x, y_pos, label='Órbita')
plt.scatter([0], [0], color='red', label='Protón')
plt.xlabel('x (Bohr)')
plt.ylabel('y (Bohr)')
plt.legend()
plt.axis('equal')
plt.title('Órbita del electrón en el potencial de Coulomb')
plt.show()

# Graficar energía
plt.figure()
plt.plot(t, energia_total, label='Energía Total')
plt.plot(t, energia_cinetica, label='Energía Cinética', linestyle='dashed')
plt.plot(t, energia_potencial, label='Energía Potencial', linestyle='dotted')
plt.xlabel('Tiempo (unidades atómicas)')
plt.ylabel('Energía (Ha)')
plt.legend()
plt.title('Energía del sistema en función del tiempo')
plt.show()

# Módulo de simulación de la órbita
def animar_orbita():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel('x (Bohr)')
    ax.set_ylabel('y (Bohr)')
    ax.set_title('Simulación de la órbita del electrón')
    ax.scatter(0, 0, color='red', label='Protón')
    electron, = ax.plot([], [], 'bo', label='Electrón')
    trayectoria, = ax.plot([], [], 'b-', alpha=0.5)
    
    def actualizar(frame):
        electron.set_data([x[frame]], [y_pos[frame]])
        trayectoria.set_data(x[:frame], y_pos[:frame])
        return electron, trayectoria
    
    anim = animation.FuncAnimation(fig, actualizar, frames=len(x), interval=20)
    plt.legend()
    plt.show()

# Ejecutar la simulación
animar_orbita()