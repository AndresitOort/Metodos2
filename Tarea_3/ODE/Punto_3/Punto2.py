import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
from scipy.stats import linregress
import os

# Función para calcular los ángulos de los afelios
def calcular_afelios(Y_2):
    num_afelios = len(Y_2)
    angulos_afelios = np.zeros(num_afelios)

    for p in range(num_afelios):
        x, y = Y_2[p][0], Y_2[p][1]
        angulos_afelios[p] = np.arctan2(y, x)  # Ángulo en radianes

    angulos_afelios = np.mod(angulos_afelios, 2 * np.pi)  # Normalizar al rango [0, 2π)
    angulos_grados = np.degrees(angulos_afelios)  # Convertir a grados
    angulos_arcsec = angulos_grados * 3600  # Convertir a segundos de arco

    return angulos_arcsec

# Función para detectar afelios (r(t) · v(t) = 0)
def event_afelio(t, Y, mu, alpha_s):
    x, y, vx, vy = Y
    return x * vx + y * vy  # Producto punto r(t) · v(t)

event_afelio.terminal = False  # No detener la simulación
event_afelio.direction = 0     # Detectar cruces en cualquier dirección

# Ecuaciones diferenciales
def ecuaciones(t, Y, mu, alpha_s):
    x, y, vx, vy = Y
    r = np.sqrt(x**2 + y**2)
    factor = -mu / r**2 * (1 + alpha_s / r**2) / r
    ax = factor * x
    ay = factor * y
    return [vx, vy, ax, ay]

# Parámetros del sistema
mu = 39.4234021  # Parámetro gravitacional
alpha_s = 1.09778201e-8  # Corrección relativista

# Condiciones iniciales
a = 0.38709893  # Semieje mayor en AU
e = 0.20563069  # Excentricidad
x0 = a * (1 + e)
y0 = 0
vx0 = 0
vy0 = np.sqrt(mu * (1 - e) / (a * (1 + e)))
Y0 = [x0, y0, vx0, vy0]

# Tiempo de simulación
t_span = (0, 10)  # 10 años
t_eval = np.linspace(0, 10, 200)  # Mayor resolución temporal

# Resolver el sistema de ecuaciones diferenciales
sol = solve_ivp(ecuaciones, t_span, Y0, args=(mu, alpha_s), 
                events=event_afelio, t_eval=t_eval, max_step=0.001)

# Extraer las soluciones
x, y, vx, vy = sol.y[0], sol.y[1], sol.y[2], sol.y[3]

# Tiempos y estados de los eventos (afelios)
event_times = sol.t_events[0]
Y_2 = sol.y_events[0]

# Calcular los ángulos de los afelios en segundos de arco
aph = calcular_afelios(Y_2)

# Ajuste lineal
slope, intercept, r_value, p_value, std_err = linregress(event_times, aph)

# Gráfica de la precesión
plt.figure()
plt.plot(event_times, aph, 'bo', label='Datos')
plt.plot(event_times, slope * np.array(event_times) + intercept, 'r-', label='Ajuste lineal')
plt.xlabel('Tiempo (años)')
plt.ylabel('Ángulo (arcsec)')
plt.title('Precesión de la órbita de Mercurio')
plt.legend()
plt.grid()

# Guardar la gráfica
save_path = "Tarea_3/ODE/Punto_3"
if not os.path.exists(save_path):
    os.makedirs(save_path)
plt.savefig(os.path.join(save_path, "3.b.pdf"), format="pdf")

# Mostrar la pendiente con su incertidumbre
print(f"Pendiente: {slope} ± {std_err} arcsec/año")

plt.show()

# Animación de la órbita
fig, ax = plt.subplots()
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("x (UA)")
ax.set_ylabel("y (UA)")
ax.set_title("Órbita de Mercurio con Corrección Relativista")

line, = ax.plot([], [], 'r-', lw=1)
dot, = ax.plot([], [], 'bo', markersize=5)

def update(frame):
    line.set_data(x[:frame+1], y[:frame+1])  
    dot.set_data([x[frame]], [y[frame]])  
    return line, dot

ani = animation.FuncAnimation(fig, update, frames=len(x), interval=10, blit=True)

# Guardar la animación
ani.save(os.path.join(save_path, "3.a.mp4"), writer="ffmpeg", fps=30)

plt.show()