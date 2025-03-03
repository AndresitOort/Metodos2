import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
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
