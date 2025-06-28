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

# Takes m, f and c and calculates prop:bg-time-schrödinger
def T_SA_APP_cont_mfc(T_c, rho, D, B, n, m, f, c):
    return T_SA_APP_cont(T_c, D, n, m)

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

    W = lambertw(np.e * rho ** (B * sqrt_n * (m - m_O1)))
    m_O2 = (np.real(W) - 1) / (math.log(rho) * B * sqrt_n)

    m_O3 = m - m_O1 - m_O2
    return T_CP_APP_cont(T_c, rho, D, B, n, m_O1, m_O2, m_O3)

# Comparing two time functions agains some x_values
def cmp_2_Tfuncs_clock(T_func1, T_func2, T_func1_label, T_func2_label, 
                       T_c, rho, D, B, n, m, f, c, 
                       x_values, x_label, yscale='linear'):
    (_, clock_values1) = T_func1(T_c, rho, D, B, n, m, f, c)
    (_, clock_values2) = T_func2(T_c, rho, D, B, n, m, f, c)
    plt.plot(x_values, clock_values1, label=T_func1_label, linestyle='--')
    plt.plot(x_values, clock_values2, label=T_func2_label)
    plt.xlabel(x_label)
    plt.ylabel(r'$T (s)$')
    plt.yscale(yscale)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    filename = f"{T_func1_label}_and_{T_func2_label}-vs-{x_label}-T_c{T_c}-rho{rho}-D{D}-B{B}-n{n}-f{f}-c{c}.pdf"
    plt.savefig("./continuous_plots/" + filename)
    plt.show()

# Plotting T_SA and T_NCP as a function of m, full fidelity
# parallelization on c cores on T_NCP
def SA_and_NCP_vs_m():
    # Constants (arbitrary but should be about reasonable scale)
    T_c = 1.0e-08 # on laptop
    rho = 2
    D   = 0.6
    B   = 0.24
    n   = 40
    m_values = np.linspace(1, 60, 500)
    f   = 1
    c   = 16

    cmp_2_Tfuncs_clock(T_SA_APP_cont_mfc, T_NCP_APP_cont_mfc, r'$T_{SA\_APP}$', r'$T_{NCP\_APP}$', 
                       T_c, rho, D, B, n, m_values, f, c, 
                       m_values, r'$m$', 'log')

def NCP_and_CP_vs_m():
    T_c = 1.0e-08
    rho = 2
    D   = 0.6
    B   = 0.24
    n   = 40
    m_values = np.linspace(15, 35, 500)
    f   = 1
    c   = 16
    cmp_2_Tfuncs_clock(T_NCP_APP_cont_mfc, T_CP_APP_cont_mfc, r'$T_{NCP\_APP}$', r'$T_{CP\_APP}$', 
                       T_c, rho, D, B, n, m_values, f, c, 
                       m_values, r'$m$', 'linear')

#SA_and_NCP_vs_m()
NCP_and_CP_vs_m()