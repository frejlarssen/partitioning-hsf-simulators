import math
import numpy as np
from scipy.special import lambertw
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 20,        # General font size
    'axes.titlesize': 20,   # Axes title size
    'axes.labelsize': 20,   # Axes labels
    'xtick.labelsize': 16,  # x tick labels
    'ytick.labelsize': 16,  # y tick labels
    'legend.fontsize': 16,  # Legend
})

# Time functions (T_) return: (coreh, clock)

# prop:bg-time-schrödinger
def T_SA_APP_cont(T_c, D, n, m):
    time = D * m * n * 2**(n) * T_c
    return (time, time)

# cor:time-ncp-continuous
def T_NCP_APP_cont(T_c, rho, D, B, n, m, c):
    sqrt_n = np.sqrt(n)
    coreh = rho ** (B*sqrt_n*m) * D * n * m * 2**(n/2) * T_c
    return (coreh, coreh / c)

# Takes m, f and c and calculates cor:time-ncp-continuous
def T_NCP_APP_cont_mfc(T_c, rho, D, B, n, m, f, c):
    (coreh_f1, clock_f1) = T_NCP_APP_cont(T_c, rho, D, B, n, m, c)
    return (coreh_f1 * f, clock_f1 * f)

# cor:time-cp-continuous
def T_CP_APP_cont(T_c, rho, D, B, n, m_O1, m_O2, m_O3):
    sqrt_n = np.sqrt(n)
    clock = (m_O1 + rho ** (B*sqrt_n*(m_O2)) * m_O2 + rho ** (B*sqrt_n*(m_O2 + m_O3)) * m_O3) * D*n * 2**(n/2) * T_c
    return (rho ** (B*sqrt_n*(m_O1)) * clock, clock)

# Takes m, f and c and calculates cor:time-cp-continuous
def T_CP_APP_cont_mfc(T_c, rho, D, B, n, m, f, c):
    sqrt_n = np.sqrt(n)
    m_O1 = math.log(c/f,rho) / (B * sqrt_n)
    m_O2 = (lambertw(np.e * rho ** (B * sqrt_n * (m - m_O1))) - 1) / (math.log(rho) * B * sqrt_n)
    m_O3 = m - m_O1 - m_O2
    return T_CP_APP_cont(T_c, rho, D, B, n, m_O1, m_O2, m_O3)


# Plotting T_SA and T_NCP as a function of m, full fidelity
# parallelization on c cores on T_NCP

# Constants (arbitrary but should be about reasonable scale)
T_c = 1.0e-08 # on laptop
rho = 2
D   = 0.6
B   = 0.24
n   = 40
#m  = 40
f   = 1
c   = 16

m_values = np.linspace(1, 60, 500)
#n_values = np.linspace(1, 80, 500)
(_, T_SA_values) = T_SA_APP_cont(T_c, D, n, m_values)
(T_NCP_values_coreh, T_NCP_values_clock) = T_NCP_APP_cont_mfc(T_c, rho, D, B, n, m_values, f, c)

plt.plot(m_values, T_SA_values, label=r'$T_{SA\_APP}$', linestyle='--')
plt.plot(m_values, T_NCP_values_clock, label=r'$T_{NCP\_APP\_clock}$')
plt.xlabel(r'$m$')
plt.ylabel(r'$T (s)$')
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.tight_layout()
filename = f"T_SA_and_NCP-vs-m-T_c{T_c}-rho{rho}-D{D}-B{B}-n{n}-f{1}-c{16}.pdf"
plt.savefig("./continuous_plots/" + filename)
plt.show()

