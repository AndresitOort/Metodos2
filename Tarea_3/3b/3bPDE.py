import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import njit, prange
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
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
