# Logística
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

L = 2
c = 1
N_x = 100 
Nt = 200

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
    siguiente_Dirichlet(frame)
    siguiente_newmann(frame)
    siguiente_periodica(frame)
    

    
    return line_dirichlet, line_newman, line_periodica

# Crear la animación
anim = animation.FuncAnimation(fig, actualizar_animacion, frames=Nt, interval=50, blit=False)

# Guardar la animación en MP4
writer = animation.FFMpegWriter(fps=30, bitrate=1800)
anim.save("animacion_ondas.mp4", writer=writer)




'''ani_dirichlet=animation.FuncAnimation(fig, siguiente_Dirichlet, frames=Nt, interval=20, blit=False)
ani_newmann = animation.FuncAnimation(fig, siguiente_newmann, frames=Nt, interval=20, blit=False)
ani_periodica = animation.FuncAnimation(fig, siguiente_periodica, frames=Nt, interval=20, blit=False)

plt.tight_layout()
plt.show()'''

