import numpy as np
import matplotlib.pyplot as plt
from numba import njit

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

# Tiempo de simulación
t_max = 10  # Debe cubrir al menos un periodo
n_pasos = 10000

t = np.linspace(0, t_max, n_pasos)
y = runge_kutta4(derivadas, y0, t)

# Extraer posiciones y velocidades
x, y_pos, vx, vy = y[:, 0], y[:, 1], y[:, 2], y[:, 3]

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

# Verificación de radio constante
radio = np.sqrt(x**2 + y_pos**2)
energia = 0.5 * (vx**2 + vy**2) - 1 / radio

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
plt.plot(t, energia, label='Energía Total')
plt.xlabel('Tiempo (unidades atómicas)')
plt.ylabel('Energía (Ha)')
plt.legend()
plt.title('Energía del sistema en función del tiempo')
plt.show()
