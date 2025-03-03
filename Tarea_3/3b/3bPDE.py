import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import njit, prange
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from scipy.special import roots_legendre
#--------------------------------------------------------------- Punto 1-----------------------------------------------------------------------
@njit(parallel=True)
def sor_iteration(phi, rho, X, Y, h, omega, max_iter, tol, phi_boundary):
    N = phi.shape[0]
    for _ in range(max_iter):
        phi_old = phi.copy()
        for i in prange(1, N-1):
            for j in prange(1, N-1):
                if X[i, j]**2 + Y[i, j]**2 < 1:
                    phi_new = 0.25 * (phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1] - h**2 * rho[i, j])
                    phi[i, j] = (1 - omega) * phi[i, j] + omega * phi_new
        
        # Aplicar condición de frontera en el círculo unitario
        for i in prange(N):
            for j in prange(N):
                if X[i, j]**2 + Y[i, j]**2 >= 1:
                    phi[i, j] = phi_boundary[i, j]
        
        # Verificar convergencia con la traza de la diferencia
        if np.trace(np.abs(phi - phi_old)) < tol:
            break
    return phi

def poisson_solver(N=1000, tol=1e-4, max_iter=15000, omega=1.9):
    L = 1.01  # Dominio ligeramente más grande que el círculo unitario
    h = 2 * L / (N - 1)  # Tamaño del paso
    x = np.linspace(-L, L, N)
    y = np.linspace(-L, L, N)
    X, Y = np.meshgrid(x, y)
    
    # Inicializar la solución con valores aleatorios dentro del dominio
    phi = np.random.rand(N, N)  # Condición inicial aleatoria
    
    # Definir la densidad de carga
    rho = np.where(X**2 + Y**2 < 1, -X-Y, 0)
    
    # Aplicar condición de frontera en el círculo unitario
    theta = np.arctan2(Y, X)
    phi_boundary = np.sin(7 * theta)
    
    # Ejecutar iteraciones con Numba
    phi = sor_iteration(phi, rho, X, Y, h, omega, max_iter, tol, phi_boundary)
    
    return X, Y, phi, phi_boundary

# Resolver el problema
X, Y, phi, phi_boundary = poisson_solver()

# # Graficar condiciones de frontera
# fig, ax = plt.subplots(figsize=(6, 6))
# circle = np.sqrt(X**2 + Y**2) < 1
# cmap = plt.cm.jet
# cmap.set_under('white')
# im = ax.imshow(np.where(circle, phi_boundary, np.nan), extent=[-1, 1, -1, 1], origin='lower', cmap=cmap)
# plt.colorbar(im, ax=ax, label='Condiciones de frontera')
# ax.set_title('Condiciones de frontera')
# plt.show()

# # Graficar la solución 3D
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, phi, cmap='jet', edgecolor='None')
# plt.title('Solución de la ecuación de Poisson')
# plt.show()

#--------------------------------------------------------------------------Punto 2-------------------------------------------------

# Logística
L = 2
c = 0.1
N_x = 150  
Nt = 10

dx = L / N_x  
dt = 1 / Nt
Courant = c * dt / dx  

lista_x = np.linspace(0, L, N_x)

# Condición inicial
def u_funcion(X, t=0):
    return np.exp(-125 * (X - 0.5) ** 2)

# Inicialización de condiciones
u_dirichlet = u_funcion(lista_x)
u_anterior_dirichlet = np.copy(u_dirichlet)
u_nuevo_dirichlet = np.zeros_like(u_dirichlet)

u_newmann = u_funcion(lista_x)
u_anterior_newmann = np.copy(u_newmann)
u_nuevo_newmann = np.zeros_like(u_newmann)

u_periodica = u_funcion(lista_x)
u_anterior_periodica = np.copy(u_periodica)
u_nuevo_periodica = np.zeros_like(u_periodica)


#---------------------------------------------Configuracion de las figuras-------------------------------------------------------------------


# Configuración de la figura con subgráficos
fig, axes = plt.subplots(3, 1, figsize=(8, 12))
(ax_dirichlet, ax_newmann, ax_periodica) = axes

# Gráficos
line_dirichlet, = ax_dirichlet.plot(lista_x, u_dirichlet, color='y',label='dirichlet')
ax_dirichlet.set_xlim(0, L)
ax_dirichlet.set_ylim(-1, 1)
ax_dirichlet.set_title("Ecuación de Onda - Condición de Dirichlet")
ax_dirichlet.legend()

line_newman, = ax_newmann.plot(lista_x, u_newmann, color='b',label='newmann')
ax_newmann.set_xlim(0, L)
ax_newmann.set_ylim(-1, 1)
ax_newmann.set_title("Ecuación de Onda - Condición de Neumann")
ax_newmann.legend()
line_periodica, = ax_periodica.plot(lista_x, u_periodica, color='r',label='periodica')
ax_periodica.set_xlim(0, L)
ax_periodica.set_ylim(-1, 1)
ax_periodica.set_title("Ecuación de Onda - Condición Periódica")
ax_periodica.legend()


#--------------------------------------------------funciones siguiente-------------------------------------------------------------------------------


#Dirichlet

def siguiente_Dirichlet(frame):
    global u_anterior_dirichlet, u_dirichlet, u_nuevo_dirichlet
    for i in range(1, N_x - 1):
        u_nuevo_dirichlet[i] = 2 * u_dirichlet[i] - u_anterior_dirichlet[i] + Courant**2 * (u_dirichlet[i+1] - 2*u_dirichlet[i] + u_dirichlet[i-1])
    
    #Condiciones de fonrtera Dirichlet
    u_nuevo_dirichlet[0], u_nuevo_dirichlet[-1] = 0, 0  
    u_anterior_dirichlet, u_dirichlet = u_dirichlet, u_nuevo_dirichlet.copy()
    line_dirichlet.set_ydata(u_dirichlet)
    return line_dirichlet,


#newmann

def siguiente_newmann(frame):
    global u_anterior_newmann, u_newmann, u_nuevo_newmann
    for i in range(1, N_x - 1):
        u_nuevo_newmann[i] = 2 * u_newmann[i] - u_anterior_newmann[i] + Courant**2 * (u_newmann[i+1] - 2*u_newmann[i] + u_newmann[i-1])
        
    #Condiciones de frontera newmann    
    u_nuevo_newmann[0] = u_nuevo_newmann[1]  
    u_nuevo_newmann[-1] = u_nuevo_newmann[-2]
    
    u_anterior_newmann, u_newmann = u_newmann, u_nuevo_newmann.copy()
    line_newman.set_ydata(u_newmann)
    return line_newman,


#periodica

def siguiente_periodica(frame):
    global u_anterior_periodica, u_periodica, u_nuevo_periodica
    for i in range(1, N_x - 1):
        u_nuevo_periodica[i] = 2 * u_periodica[i] - u_anterior_periodica[i] + Courant**2 * (u_periodica[i+1] - 2*u_periodica[i] + u_periodica[i-1])
    
    #Condiciones de frontera periódica
    u_nuevo_periodica[0] = u_nuevo_periodica[-1]  
    u_anterior_periodica, u_periodica = u_periodica, u_nuevo_periodica.copy()
    line_periodica.set_ydata(u_periodica)
    return line_periodica,

#-----------------------------------------------animaciones-----------------------------------------------------------------------------------------ani_dirichlet = animation.FuncAnimation(fig, siguiente_Dirichlet, frames=Nt, interval=20, blit=False)

def actualizar_animacion(frame):
    siguiente_Dirichlet()
    siguiente_newmann()
    siguiente_periodica()
    

    
    return line_dirichlet, line_newman, line_periodica

'''# Crear la animación
anim = animation.FuncAnimation(fig, actualizar_animacion, frames=Nt, interval=50, blit=False)

# Guardar la animación en MP4
writer = animation.FFMpegWriter(fps=30, bitrate=1800)
anim.save("animacion_ondas.mp4", writer=writer)

plt.show()'''


ani_dirichlet=animation.FuncAnimation(fig, siguiente_Dirichlet, frames=Nt, interval=20, blit=False)
ani_newmann = animation.FuncAnimation(fig, siguiente_newmann, frames=Nt, interval=20, blit=False)
ani_periodica = animation.FuncAnimation(fig, siguiente_periodica, frames=Nt, interval=20, blit=False)

plt.tight_layout()
plt.show()

#------------------------------------------------------------------------- Punto 3 --------------------------------------------------------- #

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
@njit
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
# Visualización de la convergencia de las soluciones

'''
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
# Crear animación de la solución
'''
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


#------------------------------------------------------------------------- Punto 4 ---------------------------------------------------------

plt.rcParams['animation.ffmpeg_path'] = r"C:\webm\bin\ffmpeg.exe"
plt.style.use('dark_background')
custom_cmap = LinearSegmentedColormap.from_list("my_cmap", ["blue", "black", "red"])

# Parámetros del dominio
x_max, y_max = 1.0, 2.0
Nx, Ny = 200, 400
dx, dy = x_max / Nx, y_max / Ny
dt = 0.001
T = 2.0
c0 = 0.5
clente = c0 / 5

x, y = np.linspace(0, x_max, Nx), np.linspace(0, y_max, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

c = np.full((Nx, Ny), c0)
ancho_abertura_x = 0.40
ancho_pared_y = 0.04
pared = (np.abs(Y - y_max / 2) < ancho_pared_y / 2) & ~(
    (np.abs(X - x_max / 2) < ancho_abertura_x / 2)
)
c[pared] = 0

radio_lente = ancho_abertura_x / 2
centro_x, centro_y = x_max / 2, y_max / 2
lente_region = ((X - centro_x) ** 2 + (Y - centro_y) ** 2 <= radio_lente ** 2) & (Y >= centro_y)
c[lente_region] = clente

frec = 10
fuente_x, fuente_y = 0.5, 0.5
fuente_idx = (np.abs(x - fuente_x)).argmin()
fuente_idy = (np.abs(y - fuente_y)).argmin()

u = np.zeros((Nx, Ny))
u_prev = np.zeros((Nx, Ny))
u_next = np.zeros((Nx, Ny))

cfl = (c * dt / dx) ** 2
assert np.all(cfl < 1), "El coeficiente de Courant debe ser menor que 1."

# Configurar figura
fig, ax = plt.subplots()
fig.patch.set_facecolor('black')  # Fondo negro en la figura
ax.set_facecolor('black')  # Fondo negro en los ejes

# Ajustar colores de etiquetas y ticks
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.tick_params(colors='white')

# Onda principal
cmap = ax.imshow(u.T, origin='lower', extent=[0, x_max, 0, y_max], cmap=custom_cmap, vmin=-0.01, vmax=0.01, alpha=1.0)

# Máscaras de pared y lente
pared_mask = np.zeros_like(u.T)
pared_mask[pared.T] = 1
pared_im = ax.imshow(pared_mask, origin='lower', extent=[0, x_max, 0, y_max], cmap='Greys', alpha=0.5)

lente_mask = np.zeros_like(u.T)
lente_mask[lente_region.T] = 1
lente_im = ax.imshow(lente_mask, origin='lower', extent=[0, x_max, 0, y_max], cmap='Blues', alpha=0.3)

ax.set_xlabel("X")
ax.set_ylabel("Y")

speed = 2

def update(frame):
    global u, u_prev, u_next, speed
    for _ in range(speed):
        t = frame * dt
        u_next[1:-1, 1:-1] = (2 * u[1:-1, 1:-1] - u_prev[1:-1, 1:-1] +
                              cfl[1:-1, 1:-1] * (u[2:, 1:-1] + u[:-2, 1:-1] +
                              u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]))
        u_next[fuente_idx, fuente_idy] += 0.01 * np.sin(2 * np.pi * frec * t)
        u_next[pared] = 0
        u_prev, u, u_next = u, u_next, u_prev

    cmap.set_data(u.T)
    return [cmap]

def on_key(event):
    global speed
    if event.key == "up":
        speed = min(speed + 1, 5)
    elif event.key == "down":
        speed = max(speed - 1, 1)
    print(f"Velocidad actual: {speed}x")

fig.canvas.mpl_connect("key_press_event", on_key)

ani = animation.FuncAnimation(fig, update, frames=int(T / dt), interval=20, blit=False)

# Guardar la animación en formato MP4
# writer = animation.FFMpegWriter(fps=90, metadata={"title": "Simulación de Ondas"})
# import os
# output_path = os.path.join("Tarea_3", "3b", "4.a.mp4")  # Asegura compatibilidad con Windows/Linux
# ani.save(output_path, writer=writer)

# print("Animación guardada correctamente como 4.a.mp4")
# plt.show()
