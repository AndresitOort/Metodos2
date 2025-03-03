import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from numba import njit
import matplotlib.animation as animation
from scipy.special import roots_legendre

# Parámetros
# Definimos todos los parámetros de nuestro sistema para poder simular la solución a la ecuación.
L = 2.0  # Longitud del dominio
N = 200  # La cantidad de puntos espaciales que consideraremos
dx = L / N  # Paso espacial
dt = 0.0001  # Paso temporal -> El paso temporal debe ser lo suficientemente pequeño para evitar 
             # que la solución diverja.
             
Tmax = 2  # Tiempo total
Nt = int(Tmax / dt)  # Número de pasos de tiempo -> La cantidad de pasos temporales que tendremos para
                     # el tiempo de simulación.
                     
alpha = 0.022  # Coeficiente de dispersión

# Inicialización de la malla
x = np.linspace(0, L, N, endpoint=False)
u = np.cos(np.pi * x)  # Condición inicial. 

# Función para iterar usando el esquema numérico
#@njit
def evolucion_kdv(u, dx, dt, Nt, alpha):
    Nx = len(u)
    u_hist = np.zeros((Nt, Nx)) # Creamos una malla de 0's con con Nt filas y Nx columnas
    u_hist[0] = u.copy() # Guardamos para para la fila t=0 nuestra condición inicial u ya definida. 
    
    # Primer paso con esquema modificado
    u_next = np.zeros_like(u) # Creamos un array de 0's con la misma longitud de u.
    for i in range(Nx): # Recorremos las columnas de una fila temporal
        u_next[i] = u[i] - (dt / (3 * dx)) * ( (u[np.mod(i+1, Nx)] + u[i] + u[np.mod(i-1, Nx)]) *
                                               (u[np.mod(i+1, Nx)] - u[np.mod(i-1, Nx)] ) ) \
                          - (alpha**2) * (dt / (dx**3)) * (u[np.mod(i+2, Nx)] - 2 * u[np.mod(i+1, Nx)] 
                                                          + 2 * u[np.mod(i-1, Nx)] - u[np.mod(i-2, Nx)] )
    
    u_hist[1] = u_next.copy() # Habiendo hecho esto, tenemos 2 puntos temporales donde hemos evolucionadio las
                              # Ecucaiones de KdV, con ello podemos simular los siguientes según el método del paper.

    # Iteraciones en el tiempo usando la ecuación del paper
    for t in range(1, Nt-1): # Como las condicones temporales no son periódicas, entonces hacemos la simulación 
                             # temporal dentro de la malla creada sin topar con los bordes.
        u_new = np.zeros_like(u)
        for i in range(Nx):
            u_new[i] = u_hist[t-1, i] - (dt / (3 * dx)) * ( (u_hist[t, np.mod(i+1, Nx)] + u_hist[t, i] + u_hist[t, np.mod(i-1, Nx)]) *
                                                            (u_hist[t, np.mod(i+1, Nx)] - u_hist[t, np.mod(i-1, Nx)] ) ) \
                              - (alpha**2) * (dt / (dx**3)) * (u_hist[t, np.mod(i+2, Nx)] - 2 * u_hist[t, np.mod(i+1, Nx)] 
                                                              + 2 * u_hist[t, np.mod(i-1, Nx)] - u_hist[t, np.mod(i-2, Nx)] )
        
        u_hist[t+1] = u_new.copy()

    return u_hist

# Ejecutar la simulación
solucion = evolucion_kdv(u, dx, dt, Nt, alpha)

#La imágen ya está guardada en esta carpeta

'''# Visualización del resultado
plt.figure(figsize=(20, 5))
plt.imshow(solucion.T, aspect='auto', cmap='inferno', origin='lower',
           extent=[0, Tmax, 0, L])
plt.colorbar(label='$u(x,t)$')
plt.xlabel('Tiempo')
plt.ylabel('Espacio')
plt.title('Evolución de la ecuación KdV')
plt.savefig('3a_Convergencia.pdf')
plt.show()'''

#La animación ya está guardada en esta carpeta
'''# Crear animación
fig, ax = plt.subplots(figsize=(8, 5))
line, = ax.plot(x, solucion[0], color='C0')
ax.set_xlim(0, L)
ax.set_ylim(np.min(solucion), np.max(solucion))
ax.set_xlabel('Espacio')
ax.set_ylabel('u(x,t)')
ax.set_title('Evolución de la ecuación KdV')

# Ajustar animación para durar 20 segundos
num_frames = 600  # 20 segundos * 30 fps
interval = 33.3   # 20,000 ms / 600 frames

def update(frame):
    line.set_ydata(solucion[frame * (Nt // num_frames)])  # Ajustar índice para cubrir todo Nt
    return line,

ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=True)

# Guardar la animación como mp4
ani.save('3.a.mp4', writer='ffmpeg', fps=30)
'''
def calcular_cantidades_conservadas(solucion, dx, alpha):
    n_gauss = 6  # Número de puntos de cuadratura
    xi, wi = roots_legendre(n_gauss)  # Nodos y pesos de Gauss-Legendre
    xi = 0.5 * (xi + 1)  # Transformar nodos al intervalo [0,1]
    wi = 0.5 * wi  # Ajustar los pesos

    masa = np.zeros(solucion.shape[0])
    momento = np.zeros(solucion.shape[0])
    energia = np.zeros(solucion.shape[0])

    for t in range(solucion.shape[0]):
        integral_masa = 0
        integral_momento = 0
        integral_energia = 0
        
        for i in range(N - 1):
            x_i = x[i]
            x_f = x[i + 1]
            x_gauss = x_i + (x_f - x_i) * xi  # Mapeo de nodos al subintervalo
            u_gauss = np.interp(x_gauss, x, solucion[t])  # Evaluar u en nodos
            
            integral_masa += np.sum(wi * u_gauss) * (x_f - x_i)
            integral_momento += np.sum(wi * 0.5 * u_gauss**2) * (x_f - x_i)
            derivada_u = np.gradient(u_gauss, x_gauss)
            integral_energia += np.sum(wi * ((u_gauss**3) / 3 - (alpha**2 / 2) * derivada_u**2)) * (x_f - x_i)
        
        masa[t] = integral_masa
        momento[t] = integral_momento
        energia[t] = integral_energia

    return masa, momento, energia

# Ejecutar la simulación
solucion = evolucion_kdv(u, dx, dt, Nt, alpha)

# Calcular las cantidades conservadas
masa, momento, energia = calcular_cantidades_conservadas(solucion, dx, alpha)

# Graficar cantidades conservadas
fig, axs = plt.subplots(3, 1, figsize=(8, 10))

t_vals = np.linspace(0, Tmax, len(masa))

axs[0].plot(t_vals, masa, label='Masa')
axs[0].set_ylabel('Masa')
axs[0].grid()

axs[1].plot(t_vals, momento, label='Momento')
axs[1].set_ylabel('Momento')
axs[1].grid()

axs[2].plot(t_vals, energia, label='Energía')
axs[2].set_ylabel('Energía')
axs[2].set_xlabel('Tiempo')
axs[2].grid()

#Las gráficas ya están guardadas como '3.b.pdf' dentro de esta carpeta
'''plt.tight_layout()
plt.savefig('3.b.pdf')
plt.show()'''

