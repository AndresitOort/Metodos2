import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit

# Constantes del problema
mu = 39.4234021  # UA^3 / Año^2
alpha = 1.09778201e-8  # UA^2

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
t = sol.t

# Calcular la distancia al Sol
dist = np.sqrt(x**2 + y**2)

# Detectar periastro usando derivada numérica
periastro_indices = np.argwhere((dist[1:-1] < dist[:-2]) & (dist[1:-1] < dist[2:])).flatten() + 1

# Calcular los ángulos en los puntos del periastro
theta_periastro = np.arctan2(y[periastro_indices], x[periastro_indices])

# Desenrollar manualmente los ángulos para evitar los saltos
for i in range(1, len(theta_periastro)):
    while theta_periastro[i] < theta_periastro[i-1]:
        theta_periastro[i] += 2 * np.pi

# Convertir a grados y luego a segundos de arco
theta_periastro_arcsec = np.degrees(theta_periastro) * 3600

# Ajuste lineal para determinar la precesión
coef_periastro = np.polyfit(t[periastro_indices], theta_periastro_arcsec, 1)
pendiente_periastro = coef_periastro[0] * 100  # En arcsec/siglo

# Imprimir resultados
print(f"Pendiente de la precesión del periastro: {pendiente_periastro:.4f} arcsec/siglo")

# Crear gráfico corregido
plt.figure(figsize=(12, 5))
plt.plot(t[periastro_indices], theta_periastro_arcsec, 'o-', label='Periastro')
plt.plot(t[periastro_indices], np.polyval(coef_periastro, t[periastro_indices]), 'r--', label='Ajuste lineal')
plt.xlabel('Tiempo (años)')
plt.ylabel('Ángulo (segundos de arco)')
plt.title('Precesión de la órbita de Mercurio (Periastro)')
plt.legend()
plt.text(t[periastro_indices][-1], theta_periastro_arcsec[-1], f'Pendiente: {pendiente_periastro:.4f} arcsec/siglo')
plt.grid()
plt.savefig("Tarea_3/ODE/Punto_3/3.b.pdf")