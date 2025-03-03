import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import matplotlib.animation as animation

#-----------------------------------------------------------------------------Punto 2.a

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
x, y_pos, vx, vy = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
y_sin_larmor = y_pos
# Calcular el período simulado
"""
detecta los momentos en los que la coordenada y
y del electrón cambia de signo, lo que indica que ha cruzado el eje horizontal. 
Para hacerlo, primero usa np.sign(y_pos) para obtener si cada valor de 
y es positivo, negativo o cero, luego np.diff encuentra dónde hay un cambio entre estos signos,
y finalmente np.where[0] devuelve los índices de esos cambios. Multiplicamos por 2 pues lo hacemos con el eje y.
"""
cambios_signo = np.where(np.diff(np.sign(y_pos)))[0]
if len(cambios_signo) > 1:
    P_sim = 2 * (t[cambios_signo[1]] - t[cambios_signo[0]])
else:
    P_sim = np.nan  # Si no hay suficiente precisión, no se calcula

# Convertir a attosegundos (h)
P_teo_as = T_teo * 5.243e7
P_sim_as = P_sim * 5.243e7

# Imprimir resultado
print(f"2.a) {{P_teo = {P_teo_as:.5f}; P_sim = {P_sim_as:.5f}}} attosegundos")

#Graficamos
# Verificación de radio y energía constantes
radio = np.sqrt(x**2 + y_pos**2)
energia_cinetica = 0.5 * (vx**2 + vy**2)
energia_potencial = -1 / radio
energia_total = energia_cinetica + energia_potencial
"""
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
"""
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
plt.close()

# Graficar energía
plt.figure()
plt.plot(t, energia_total, label='Energía Total')
plt.plot(t, energia_cinetica, label='Energía Cinética', linestyle='dashed')
plt.plot(t, energia_potencial, label='Energía Potencial', linestyle='dotted')
plt.xlabel('Tiempo (unidades atómicas)')
plt.ylabel('Energía (Ha)')
plt.legend()
plt.title('Energía del sistema en función del tiempo')
plt.savefig("2.a.Verificacion_energía_constante.pdf")
plt.show()
plt.close()
# Módulo de simulación de la órbita
"""
Para animar la órbita del electrón, primero necesitamos un objeto de la clase Figure, que representa nuestro lienzo,
y un objeto Axes, que define la región dentro de la figura donde dibujaremos. Configuramos las características de ax 
para que los ejes permanezcan fijos en cada cuadro y no se reajusten dinámicamente.

A continuación, creamos los objetos electron y trayectoria, que son gráficos (plot). 
Inicialmente, estos se definen como gráficos vacíos, ya que su contenido se actualizará en cada fotograma mediante la función
actualizar.

La función FuncAnimation gestiona la actualización automática de estos elementos, 
ejecutando de manera recursiva la función actualizar para cada cuadro de la animación, 
lo que permite visualizar la evolución del sistema en el tiempo.
"""


def animar_orbita(nombre_archivo):
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
        electron.set_data([x[frame]], [y_pos[frame]]) #Recordemos que tiene que recibir un []
        trayectoria.set_data(x[:frame], y_pos[:frame])
        return electron, trayectoria
    
    anim = animation.FuncAnimation(fig, actualizar, frames=len(x), interval=20)
    writer = animation.PillowWriter(fps=30)
    anim.save(nombre_archivo, writer=writer)

    plt.legend()
    plt.close()

# Ejecutar la simulación
animar_orbita("2.a.animation.gif")

#--------------------------------------------------------------------------------Punto 2.b

# Constante de estructura fina
alpha = 1 / 137
#Ya tenemos una función que, dado un vector de posición nos calcula la fuerza de coulomb asociada
#Tenemos una función que nos pasa de [x0, y0, vx0, vy0] a [x1, y1, vx1, vy1].

#Implementamos el Runge Kutta modificado. Para esto, primero, vemos los pasos del runge
def derivadas_larmor(t,y): #Como el runge kutta evalua en 4 puntos del intervalo, necesitamos que la aceleración también se evalúe

    x, y, vx, vy, a = y

    r = np.sqrt(x**2 + y**2)

    ax = -x/np.abs(r**3)
    ay = -y/np.abs(r**3)

    a_m = np.sqrt(ax**2 + ay**2)

    return np.array([vx,vy,ax,ay,a_m])

def runge_kutta4_larmor(f, y0, t):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    dt = t[1] - t[0]
    indice_final = 0
    for i in range(n - 1):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + dt/2, y[i] + dt*k1/2)
        k3 = f(t[i] + dt/2, y[i] + dt*k2/2)
        k4 = f(t[i] + dt, y[i] + dt*k3)
        y_coulomb = y[i] + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        #Hasta acá, tenemos nuestro vector asociado a coulomb [x1, y1, vx1, vy1]
        #Asumimos que la velocidad sobre la que trabaja larmor es la que arroja coulomb
        v_temporal = np.array([y_coulomb[2],y_coulomb[3]])
        v_temporal_norma = np.linalg.norm(v_temporal)
        v_temporal_unitario = v_temporal/v_temporal_norma
        #Asumimos que la aceleración que nos interesa es la asociada a la posición [x0, y0]
        a_norm = y_coulomb[4]
        #Sacamos el factor de corrección
        factor = np.sqrt(max((v_temporal_norma**2)- ((4/3) * (alpha**3) * (a_norm**2) * dt),0))
        if factor>0:
            v_real = v_temporal_unitario * factor
            y_coulomb[2] = v_real[0]
            y_coulomb[3] = v_real[1]
        if 0.01<y[i][0]:
            indice_final = i
        y[i+1] = y_coulomb


        #Nuestro vector queda de la forma  v_temporal_unitario*factor de corrección
    return y, indice_final
t_max = 10000
n_pasos = 100000
y0 = np.append(y0, 0)
t = np.linspace(0, t_max, n_pasos)
y, indice_final = runge_kutta4_larmor(derivadas_larmor, y0, t)

# Extraer posiciones y velocidades
x, y_pos, vx, vy = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
x = x[:indice_final]
y_pos = y_pos[:indice_final]
vx = vx[:indice_final]
vy = vy[:indice_final]

radio = np.sqrt(x**2 + y_pos**2)
energia_cinetica = 0.5 * (vx**2 + vy**2)
energia_potencial = -1 / radio
energia_total = energia_cinetica + energia_potencial

t_sol = t[:indice_final] 
t_fall_as = t_sol[-1]* 5.23e+7
print(f"2.b) {{t_fall = {t_fall_as:.5f}}} attosegundos")

#Creamos los pdfs:
plt.figure(figsize=(6,6))
plt.plot(x, y_pos, label="Órbita del electrón")
plt.scatter(0, 0, color="red", label="Protón")
plt.xlabel("x (Bohr)")
plt.ylabel("y (Bohr)")
plt.axis("equal")
plt.legend()
plt.title("Órbita del electrón")
plt.savefig("2.b.XY.pdf")
plt.close()

# Graficar diagnósticos (2.b.diagnostics.pdf)
fig, axs = plt.subplots(3, 1, figsize=(8,10))

axs[0].plot(t_sol, energia_total[:indice_final], label="Energía Total")
axs[0].set_xlabel("Tiempo (u.t.)")
axs[0].set_ylabel("Energía Total (Ha)")
axs[0].legend()
axs[0].grid()

axs[1].plot(t_sol, energia_cinetica[:indice_final], label="Energía Cinética", color="orange")
axs[1].set_xlabel("Tiempo (u.t.)")
axs[1].set_ylabel("Energía Cinética (Ha)")
axs[1].legend()
axs[1].grid()

axs[2].plot(t_sol, radio[:indice_final], label="Radio", color="green")
axs[2].set_xlabel("Tiempo (u.t.)")
axs[2].set_ylabel("Radio (Bohr)")
axs[2].legend()
axs[2].grid()

plt.tight_layout()
plt.savefig("2.b.diagnostics.pdf")
plt.close()

animar_orbita("2.b.animation.gif")

