import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

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
P_teo_as = T_teo * 24.16
P_sim_as = P_sim * 24.16

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
#animar_orbita("2.a.animation.gif")

#--------------------------------------------------------------------------------Punto 2.b

# Constante de estructura fina
alpha = 1 / 137
#Ya tenemos una función que, dado un vector de posición nos calcula la fuerza de coulomb asociada
#Tenemos una función que nos pasa de [x0, y0, vx0, vy0] a [x1, y1, vx1, vy1].

#Implementamos el Runge Kutta modificado. Para esto, primero, vemos los pasos del runge

@njit
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
        a = fuerza_coulomb(np.array([y[i][0], y[i][1]]))
        a_norm = np.linalg.norm(a)
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

t_max = 10000000
n_pasos = 90000000
t = np.linspace(0, t_max, n_pasos)
y, indice_final = runge_kutta4_larmor(derivadas, y0, t)

# Extraer posiciones y velocidades
x, y_pos, vx, vy = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
x = x[:indice_final:]
y_pos = y_pos[:indice_final]
vx = vx[:indice_final]
vy = vy[:indice_final]

#Sacamos las energías
"""
radio = np.sqrt(x**2 + y_pos**2)
energia_cinetica = 0.5 * (vx**2 + vy**2)
energia_potencial = -1 / radio
energia_total = energia_cinetica + energia_potencial
"""

t_sol = t[:indice_final] 
t_fall_as = t_sol[-1]* 24.16
print(f"2.b) {{t_fall = {t_fall_as:.5f}}} attosegundos")

#Gráficas 
"""
#Animamos y guardamos las gráficas
animar_orbita("2.b.animation.gif")

plt.figure(figsize=(6,6))
plt.plot(x, y_pos, label="Órbita del electrón")
plt.scatter(0, 0, color="red", label="Protón")
plt.xlabel("x (Bohr)")
plt.ylabel("y (Bohr)")
plt.axis("equal")
plt.legend()
plt.title("Órbita del electrón")
plt.savefig("2.b.XY.pdf")
plt.show()
plt.close()

# Graficar diagnósticos (2.b.diagnostics.pdf)
fig, axs = plt.subplots(3, 1, figsize=(8,10))

axs[0].plot(t_sol[::10000], energia_total[:indice_final:10000], label="Energía Total")
axs[0].set_xlabel("Tiempo (u.t.)")
axs[0].set_ylabel("Energía Total (Ha)")
axs[0].legend()
axs[0].grid()

axs[1].plot(t_sol[::10000], energia_cinetica[:indice_final:10000], label="Energía Cinética", color="orange")
axs[1].set_xlabel("Tiempo (u.t.)")
axs[1].set_ylabel("Energía Cinética (Ha)")
axs[1].legend()
axs[1].grid()

axs[2].plot(t_sol[::10000], radio[:indice_final:10000], label="Radio", color="green")
axs[2].set_xlabel("Tiempo (u.t.)")
axs[2].set_ylabel("Radio (Bohr)")
axs[2].legend()
axs[2].grid()

plt.tight_layout()
plt.savefig("2.b.diagnostics.pdf")
plt.show()
plt.close()

"""

#---------------------------------------------------------------------Punto 4

#Lógica del Programa
"""
Queremos encontrar las energías para las cuales la solución del oscilador armónico no diverge. 
Sabemos que, en la práctica, esas energías siguen la forma 
ℏω(n+0.5). Aquí nos basamos en que las soluciones que satisfacen las condiciones de frontera son justamente aquellas
 para las que 𝜓(𝑥max)=0 en el borde del potencial. Las que no cumplan esto se descartan.

Ahora bien, ¿cómo distinguimos cuáles convergen y cuáles divergen? Definimos un criterio de divergencia según el cual, 
si ∣𝜓(𝑥max)∣ es muy grande, consideramos que la solución diverge. El signo de 𝜓(𝑥max) también resulta importante: 
entre dos soluciones convergentes, típicamente se intercalan soluciones que divergen. 
A medida que aumentamos la energía, la función puede alcanzar valores altos pero después “bajar” hasta cero (o viceversa, 
si diverge en sentido negativo). Esto hace que, en el borde, el comportamiento se asemeje a un patrón “sinusoidal”: 
hallamos un cero, luego una divergencia, después la función debe volver a descender para alcanzar otro cero, y así sucesivamente.

Por este motivo, nos interesa detectar cambios de signo y aplicar el teorema del valor intermedio para encontrar,
mediante el método de bisección, los ceros de 𝜓(𝑥max) en función de la energía. Esos ceros corresponden precisamente a las
energías físicas permitidas.
"""
# =====================================================
# 1) Definimos la EDO de Schrödinger para el oscilador
# =====================================================

# Ecuación: psi''(x) = (x^2 - 2E)*psi(x)
# Pasamos a primer orden:
#   y0 = psi,  y1 = psi'
#   => dy0/dx = y1
#      dy1/dx = (x^2 - 2E)*y0


#x los valores a evaluar (de 0 a 6);y nuestro vector de estado de la forma (x0, v0), Y E es la energía a barrer.
def ho_equation(x, y, E):
    return [y[1], (x**2 - 2*E)*y[0]]

# =====================================================
# 2) Integramos con solve_ivp (LSODA) desde x=0 a x_max
#    Condiciones iniciales según la paridad
# =====================================================
def integrate_ho(E, parity, x_max=5.0, n_points=1000, max_amp=1e6):
    """
    Integra la ecuación ho_equation para una energía E y una paridad dada. Cada paridad tiene asociada condiciones iniciales
    específicas, pues, parecido a un seno o coseno, cuando x=0 -> coseno(0)=1, y su derivada es 0, y visceversa con el seno.
 
    - parity: "even"  => psi(0)=1,  psi'(0)=0
              "odd"   => psi(0)=0,  psi'(0)=1
    - x_max: límite de integración en x>0
    - n_points: puntos en la malla
    - max_amp: umbral para decidir si la función diverge

    Retorna:
      sol.t     -> array de x
      sol.y[0]  -> array de psi(x)
      sol.y[1]  -> array de psi'(x)
      status    -> 0 si ok, 1 si divergió
    """
    if parity == "even":
        y0 = [1.0, 0.0]  # psi(0)=1, psi'(0)=0
    elif parity == "odd":
        y0 = [0.0, 1.0]  # psi(0)=0, psi'(0)=1
    else:
        raise ValueError("parity debe ser 'even' o 'odd'")
    
    #Qué es shooting?
    """
    Cuando decimos que un “evento se dispara” en el contexto de la integración con solve_ivp, 
    queremos decir que el integrador ha detectado que la función de evento ha cruzado cero 
    (en la dirección que uno le especifica). Es decir, “disparar” (o “trigger”) es la forma coloquial de decir 
    que el evento se ha cumplido y que solve_ivp ha “tomado nota” o “actuado” en consecuencia (El terminal, por ejemplo, 
    indica que la acción a seguir es parar).
    """

    #Por qué podemos agregarle esos métodos tan raros a nuestra función?
    """
    Ponemos la condicion de que la solve_ivp pare la integración cuando el valor de posicion y[0] supere a la amplitud
    Máxima permitida, es decir, que max_amp - np.abs(y[0]) sea negativo. Acá la funcion la tratamos como un
    Objeto. Python permite crear atributos arbitrarios para ciertos tipos de objetos con una sintaxis específica, y
    resulta que las funciones tienen incluidas ese tipo de sintaxis dentro:
    de https://stackoverflow.com/questions/2280334/shortest-way-of-creating-an-object-with-arbitrary-attributes-in-python:
    obj = type('', (), {})()
    obj.hello = "hello"
    obj.world = "world"
    print obj.hello, obj.world   # will print "hello world"
    vemos que aqui los objetos definidos de dicha manera permiten agregarles atributos arbitrarios. Aquí se le agregan
    atributos con nombres específicos que Scipy sabe cómo trabajar. Acá se usan los atributos terminal y direction
    para scipy pueda entender lo siguiente:
    .terminal: si es True, el evento se considera “terminal”, es decir, cuando la función cruza cero,
     el integrador detiene la integración.
    .direction: indica si el evento se dispara en cruces de cero con pendiente negativa (-1), pendiente positiva (+1), o ambas (0).
    El direction es necesario pues, si pasa de negativo a positivo, indicaría simplemente que la energía con 
    la que comenzamos es mayor que el umbral, pero que se está disminuyendo, es decir, que no diverge la solución.
    En nuestro caso solo nos interesa que vaya de subiendo.
    """

    def event_diverge(x, y):
        # Evento para detener si |psi| supera max_amp
        return max_amp - np.abs(y[0])
    event_diverge.terminal = True
    event_diverge.direction = -1

    x_eval = np.linspace(0, x_max, n_points)
    sol = solve_ivp(
        fun=lambda xx, yy: ho_equation(xx, yy, E), 
        #fun debe recibir una funcion con una signatura específica fun(t, y) -> dydt. Como la nuestra tiene 3
        #Usamos lambda para que fun entienda que solo puede cambiar xx, yy (no E). Es decir, para que solve_ivp lo vea como
        #constante
        t_span=(0, x_max),
        y0=y0,
        t_eval=x_eval,
        method="LSODA", #Punto curioso: Este método tiene control de paso, es decir, tiene implementaciones para saber
                        #si el error asociado es muy grande
        events=event_diverge
    ) 

    # status=0 => no diverge, status=1 => divergió
    """
    En SciPy, cuando usas solve_ivp, el objeto que te devuelve (del tipo OdeResult) 
    incluye un atributo status que indica la razón por la que la integración terminó. Por convención:
    ==0: Integración exitosa
    ==1: Se encontró un terminal
    ==-1: Se encontró un fallo
    """
    status = 0 if sol.status == 0 else 1
    return sol.t, sol.y[0], sol.y[1], status 
    #Recordemos que solve_ivp entrega un objeto. y[0] devuelve la solucion para psi(x), y y[1] sus derivadas.
    #Si ponemos dense_output, ese y[0] ya viene interpolado, y lo podemos evaluar en cualquier punto



# =====================================================
# 3) Función de disparo: valor de psi(x_max) (o divergencia)
# =====================================================
def shooting(E, parity, x_max=5.0):
    """
    Integra y retorna psi(x_max).
    Si diverge, retorna un valor que se pasa del maximo estupilado con signo de psi al final.
    """
    t, psi, dpsi, st = integrate_ho(E, parity, x_max=x_max)
    if st == 1:  # divergencia
        # Tomar el signo de psi en el último punto, para identificar luego los cambios de signos y usar biseccion
        #para hallar el 0 que está entre ellos. No se usa el valor específico para que la busqueda sea más rápida
        return np.sign(psi[-1]) * 1e6
    else:
        return psi[-1]

# =====================================================
# 4) Encontrar energías permitidas por bisección
#    (buscando ceros de shooting(E))
# =====================================================
def find_eigenvalues(parity, E_min, E_max, n_eigs=5, x_max=5.0, steps=300):
    """
    Escanea en el rango [E_min, E_max] con 'steps' puntos.
    Busca sign changes en shooting(E) para aislar ceros
    y usa brentq para refinar la raíz.
    Devuelve lista de energías (float).

    ¿Cómo funciona brentq?

    """
    E_vals = np.linspace(E_min, E_max, steps)
    f_vals = [shooting(E, parity, x_max) for E in E_vals]

    eigenvals = []
    for i in range(len(E_vals)-1):
        if f_vals[i]*f_vals[i+1] < 0:  # Identificamos los cambios de signo respecto al siguiente valor
            E_left, E_right = E_vals[i], E_vals[i+1]
            # Buscamos el 0 entre ellos de manera más precisa
            E_root = brentq(lambda EE: shooting(EE, parity, x_max), E_left, E_right) #Le entregamos la función, y el intervalo
            #en el que va a buscar. Brentq funciona solo para funciones con cambio de signo.
            eigenvals.append(E_root)
            if len(eigenvals) >= n_eigs:
                break
    return eigenvals

# =====================================================
# 5) Reintegrar, reflejar y normalizar la función de onda
# =====================================================
def get_wavefunction(E, parity, x_max=5.0, n_points=1000):
    """
    Integra la ecuación en x>=0, refleja según la paridad y normaliza.
    Retorna x_full, psi_full (ordenados de -x_max a x_max).
    """
    t_pos, psi_pos, dpsi_pos, st = integrate_ho(E, parity, x_max, n_points)
    # Reflejar:
    #  even => psi(-x)= +psi(x)
    #  odd  => psi(-x)= -psi(x)
    if parity == "even":
        psi_neg = psi_pos[::-1]
    else:
        psi_neg = -psi_pos[::-1]

    x_neg = -t_pos[::-1]
    x_full = np.concatenate([x_neg, t_pos])
    psi_full = np.concatenate([psi_neg, psi_pos])

    # Ordenar de menor a mayor x (solo por precaución)
    idx = np.argsort(x_full)
    x_full = x_full[idx]
    psi_full = psi_full[idx]

    # Normalizar (trapecio simple)
    dx = x_full[1] - x_full[0]
    norm = np.sqrt(np.sum(psi_full**2)*dx)
    psi_full /= norm

    return x_full, psi_full

# =====================================================
# 6) Calcular los primeros 5 estados pares e impares
# =====================================================
x_max = 5.0
n_eigs = 5

# Rango de energías a explorar
E_min, E_max = 0.0, 15.0

even_eigs = find_eigenvalues("even", E_min, E_max, n_eigs, x_max)
odd_eigs  = find_eigenvalues("odd",  E_min, E_max, n_eigs, x_max)

print("Energías pares (even):", even_eigs)
print("Energías impares (odd):", odd_eigs)

# =====================================================
# 7) Graficar las funciones de onda "apiladas" + potencial
# =====================================================
plt.figure(figsize=(6,6))

# Graficamos el potencial V(x)=0.5*x^2
x_plot = np.linspace(-x_max, x_max, 400)
V_plot = 0.5*x_plot**2
plt.plot(x_plot, V_plot, '--', color='gray', alpha=0.5)

# Juntamos ambas listas de energías
all_eigs = []
# Intercalamos: E0(even), E0(odd), E1(even), E1(odd), ...
for ev, od in zip(even_eigs, odd_eigs):
    all_eigs.append((ev, "even"))
    all_eigs.append((od, "odd"))

# Si quieres listarlas en orden ascendente total, haz un sort. 
# Pero aquí las intercalamos por pares. 
# Para un diagrama más "ordenado" en la vertical, a veces se prefiere 
#  simplemente plotear las pares primero y luego las impares.

colors = plt.cm.rainbow(np.linspace(0,1,len(all_eigs)))
for i, (E_val, parity) in enumerate(all_eigs):
    # Obtener la función de onda normalizada
    x_sol, psi_sol = get_wavefunction(E_val, parity, x_max, 1000)
    # Escalamos la amplitud y la desplazamos a la altura E_val
    offset = E_val
    amp = 0.4  # factor para que las oscilaciones se vean
    psi_offset = offset + amp*psi_sol
    plt.plot(x_sol, psi_offset, color=colors[i], label=f"{parity}, E={E_val:.3f}")
"""
plt.ylim(0, max(odd_eigs+even_eigs)+1)
plt.xlim(-x_max, x_max)
plt.xlabel("x")
plt.ylabel("Energía")
plt.title("Oscilador armónico cuántico (método de disparo con solve_ivp)")
plt.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig("4.pdf")
plt.show()
"""
