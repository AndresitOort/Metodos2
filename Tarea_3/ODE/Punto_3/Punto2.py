import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
from numba import njit

# Constantes del problema
mu = 39.4234021  # UA^3 / Año^2
alpha = 1.09778201e-2  # UA^2

# Condiciones iniciales
a = 0.38709893  # Semieje mayor (UA)
e = 0.20563069  # Excentricidad
x0 = a * (1 + e)
y0 = 0
vx0 = 0
vy0 = np.sqrt(mu / a) * np.sqrt((1 - e) / (1 + e))

# Definición del sistema de ecuaciones diferenciales con Numba
@njit
def equations(t, Y):
    x, y, vx, vy = Y
    r2 = x**2 + y**2
    r = np.sqrt(r2)
    factor = mu / (r2 * r) * (1 + alpha / r2)
    ax = -factor * x
    ay = -factor * y
    return np.array([vx, vy, ax, ay])

# Tiempo de simulación (10 años)
t_span = (0, 10)
t_eval = np.linspace(0, 10, 1000)

# Resolver la ecuación diferencial con parámetros optimizados
sol = solve_ivp(equations, t_span, [x0, y0, vx0, vy0], method='DOP853', t_eval=t_eval, rtol=1e-10, atol=1e-12)

# Extraer soluciones
x, y = sol.y[0], sol.y[1]

# Crear animación
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlabel('x (UA)')
ax.set_ylabel('y (UA)')
ax.set_title('Órbita de Mercurio con corrección relativista')
ax.set_aspect('equal')
ax.set_xlim(-0.5, 0.5)  # Ajusta estos valores según el tamaño deseado
ax.set_ylim(-0.5, 0.5)
ax.scatter([0], [0], color='red', label='Sol')
ax.legend()
trajectory, = ax.plot(x[:1], y[:1], '-', color='gray')  # Inicializar con un solo punto
mercury, = ax.plot([], [], 'o', color='blue', label='Mercurio')

# Función de actualización de la animación
def update(frame):
    trajectory.set_data(x[:frame+1], y[:frame+1])  # Trazo progresivo de la órbita
    mercury.set_data([x[frame]], [y[frame]])
    return trajectory, mercury

ani = animation.FuncAnimation(fig, update, frames=len(x), interval=20)
ani.save("Tarea_3/ODE/Punto_3/3.a.mp4", fps=30)
