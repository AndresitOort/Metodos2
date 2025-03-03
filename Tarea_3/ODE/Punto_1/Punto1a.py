from numba import njit
import numpy as np
import matplotlib.pyplot as plt

g = 9.773
m = 10
v_0 = 10
dt = 0.001
N_MAX = 10000  

@njit
def ODEX(t, Y, b):
    x, y, vx, vy = Y
    v = np.hypot(vx**2, vy**2)
    dvx_dt = - (b/m) * vx * v
    dvy_dt = -g - (b/m) * vy * v
    return np.array([vx, vy, dvx_dt, dvy_dt])

@njit
def RK4(F, t0, Y0, dt, b):
    t = t0
    x_list = np.empty(N_MAX)
    y_list = np.empty(N_MAX)
    wrea = 0

    x_list[0], y_list[0] = Y0[0], Y0[1]
    i = 0

    while Y0[1] >= 0 and i < N_MAX - 1:
        k1 = F(t, Y0, b)
        k2 = F(t + dt / 2, Y0 + dt * k1 / 2, b)
        k3 = F(t + dt / 2, Y0 + dt * k2 / 2, b)
        k4 = F(t + dt, Y0 + dt * k3, b)

        Y0n = Y0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        dx, dy = Y0n[0] - Y0[0], Y0n[1] - Y0[1]
        ds = np.hypot(dx**2,dy**2)
        mvs = np.hypot(Y0n[2]**2, Y0n[3]**2)
        wrea += b * (mvs**2) * ds

        x_list[i+1] = Y0n[0]
        y_list[i+1] = Y0n[1]

        Y0 = Y0n
        t += dt
        i += 1

    return x_list[:i], y_list[:i], wrea

def FBA(b):
    angles = np.linspace(0, 90, 501)
    best_angle = 0
    max_range = 0

    for angle in angles:
        rad_angle = np.radians(angle)
        v0x, v0y = v_0 * np.cos(rad_angle), v_0 * np.sin(rad_angle)
        x, _, _ = RK4(ODEX, 0, np.array([0, 0, v0x, v0y]), dt, b)

        if x[-1] > max_range:
            max_range = x[-1]
            best_angle = angle

    fine_angles = np.linspace(best_angle - 0.15, best_angle + 0.15, 501)

    for angle in fine_angles:
        rad_angle = np.radians(angle)
        v0x, v0y = v_0 * np.cos(rad_angle), v_0 * np.sin(rad_angle)
        x, _, _ = RK4(ODEX, 0, np.array([0, 0, v0x, v0y]), dt, b)

        if x[-1] > max_range:
            max_range = x[-1]
            best_angle = angle

    hyperfine_angles = np.linspace(best_angle - 0.002, best_angle + 0.002, 501)

    for angle in hyperfine_angles:
        rad_angle = np.radians(angle)
        v0x, v0y = v_0 * np.cos(rad_angle), v_0 * np.sin(rad_angle)
        x, _, _ = RK4(ODEX, 0, np.array([0, 0, v0x, v0y]), dt, b)

        if x[-1] > max_range:
            max_range = x[-1]
            best_angle = angle

    return max_range, best_angle




b = 0
mr, ba = FBA(b)
x, y, wrea = RK4(ODEX, 0, np.array([0, 0, v_0 * np.cos(np.radians(ba)), v_0 * np.sin(np.radians(ba))]), dt, b)

print(f"El ángulo óptimo es {ba:.4f}°")
print(f"Alcance máximo: {mr:.4f}")
print(f"Energía disipada: {wrea:.4f}")

beta = np.logspace(-3, np.log10(2), 100)
thetam = np.zeros(len(beta))
wream = np.zeros(len(beta))

for i in range(len(beta)):
    max_range, best_angle = FBA(beta[i]) 
    thetam[i] = best_angle  

    v0x, v0y = v_0 * np.cos(np.radians(best_angle)), v_0 * np.sin(np.radians(best_angle))
    _, _, wrea = RK4(ODEX, 0, np.array([0, 0, v0x, v0y]), dt, beta[i])
    wream[i] = wrea 


save_path = r"Tarea_3\ODE\Punto_1"

plt.figure()
plt.xscale("log")
plt.plot(beta, thetam, marker='o', linestyle='-', color='b', label=r'$\theta_{\max}$ vs $\beta$')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\theta_{\max}$ (°)')
plt.legend()
plt.grid()
plt.savefig(f"{save_path}\\1.a.pdf", format="pdf")
plt.close()

plt.figure()
plt.xscale("log")
plt.plot(beta, wream, marker='s', linestyle='-', color='r', label=r'$\Delta E$ vs $\beta$')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\Delta E$')
plt.legend()
plt.grid()
plt.savefig(f"{save_path}\\1.b.pdf", format="pdf")
plt.close()