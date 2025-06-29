import math
import numpy as np
from scipy.special import lambertw
import matplotlib.pyplot as plt
import re

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

# time-cp-continuous_litterature original
def T_CP_APP_cont_litterature(T_c, rho, D, B, n, m_O1, m_O2, m_O3):
    sqrt_n = np.sqrt(n)
    n_const = 40
    C_SFA_inv = D*n_const*T_c
    clock = rho ** (B*sqrt_n*(m_O2 + m_O3)) * 2**(n/2+1) * C_SFA_inv
    return (rho ** (B*sqrt_n*(m_O1)) * clock, clock)

# Takes m, f and c and calculates cor:time-cp-continuous
def T_CP_APP_cont_mfc(T_c, rho, D, B, n, m, f, c):
    sqrt_n = np.sqrt(n)
    m_O1 = math.log(c/f,rho) / (B * sqrt_n)

    W = lambertw(np.e * rho ** (B * sqrt_n * (m - m_O1)))
    m_O2 = (np.real(W) - 1) / (math.log(rho) * B * sqrt_n)

    m_O3 = m - m_O1 - m_O2
    return T_CP_APP_cont(T_c, rho, D, B, n, m_O1, m_O2, m_O3)

# Takes m, f and c and calculates cor:time-cp-continuous
def T_CP_APP_cont_mfc_suff(T_c, rho, D, B, n, m, f, c):
    sqrt_n = np.sqrt(n)
    m_O1 = math.log(c/f,rho) / (B * sqrt_n)

    W = lambertw(np.e * rho ** (B * sqrt_n * (m - m_O1)))
    m_O2 = (np.real(W) - 1) / (math.log(rho) * B * sqrt_n)

    m_O3 = m - m_O1 - m_O2
    return T_CP_APP_cont_suff(T_c, rho, D, B, n, m_O1, m_O2, m_O3)

def T_CP_APP_cont_mfc_litterature(T_c, rho, D, B, n, m, f, c):
    sqrt_n = np.sqrt(n)
    m_O1 = math.log(c/f,rho) / (B * sqrt_n)

    W = lambertw(np.e * rho ** (B * sqrt_n * (m - m_O1)))
    m_O2 = (np.real(W) - 1) / (math.log(rho) * B * sqrt_n)

    m_O3 = m - m_O1 - m_O2
    return T_CP_APP_cont_litterature(T_c, rho, D, B, n, m_O1, m_O2, m_O3)


def sanitize_label(label):
    return re.sub(r'[^a-zA-Z0-9_-]', '', label)

# Comparing up to four functions agains some x_values
def cmp_Tfuncs_clock(T_c, rho, D, B, n, m, f, c, 
                       x_values, x_label, 
                       T_func1, T_func1_label, T_func2, T_func2_label, 
                       T_func3=None, T_func3_label=None, T_func4=None, T_func4_label=None,
                       yscale='linear'):
    (_, clock_values1) = T_func1(T_c, rho, D, B, n, m, f, c)
    plt.plot(x_values, clock_values1, label=T_func1_label, linestyle='--')
    (_, clock_values2) = T_func2(T_c, rho, D, B, n, m, f, c)
    plt.plot(x_values, clock_values2, label=T_func2_label, linestyle='-.')
    if (T_func3):
        (_, clock_values3) = T_func3(T_c, rho, D, B, n, m, f, c)
        plt.plot(x_values, clock_values3, label=T_func3_label, linestyle=':')
    if (T_func4):
        (_, clock_values4) = T_func4(T_c, rho, D, B, n, m, f, c)
        plt.plot(x_values, clock_values4, label=T_func4_label, linestyle='-')
    plt.xlabel(x_label)
    plt.ylabel(r'$T (s)$')
    plt.yscale(yscale)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    filename = f"{sanitize_label(T_func1_label)}_and_{sanitize_label(T_func2_label)}"
    if T_func3:
        filename += f"_and_{sanitize_label(T_func3_label)}"
    if T_func4:
        filename += f"_and_{sanitize_label(T_func4_label)}"
    filename += f"-vs-{sanitize_label(x_label)}-T_c{T_c}-rho{rho}-D{D}-B{B}"

    if not isinstance(n, np.ndarray):
        filename += f"-n{n}"
    if not isinstance(m, np.ndarray):
        filename += f"-m{m}"
    filename += f"-f{f}-c{c}.pdf"

    plt.savefig("./continuous_plots/" + filename)
    plt.show()

def NCP_vs_mO2():
    # Constants (arbitrary but should be about reasonable scale)
    T_c = 1.0e-08 # on laptop
    rho = 2
    D   = 0.6
    B   = 0.24
    n   = 40
    m   = 20
    f   = [0.08, 0.2,0.4,0.6,0.8,1.0]
    c   = 16

    sqrt_n = np.sqrt(n)
    m_O1_list = [math.log(c / fi, rho) / (B * sqrt_n) for fi in f]

    plt.figure(figsize=(10, 6))

    for i, (fi, m_O1) in enumerate(zip(f, m_O1_list)):
        m_O2 = np.linspace(1, m - m_O1, 500)
        m_O3 = m - m_O1 - m_O2
        (_, clock_values) = T_CP_APP_cont(T_c, rho, D, B, n, m_O1, m_O2, m_O3)
        plt.plot(m_O2, clock_values, label=fr'$f={fi}$', linestyle='--')

    plt.xlabel(r'$m_{\Omega_2}$')
    plt.ylabel(r'$T (s)$')
    plt.yscale('linear')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    filename = f"T_CP_APP-vs-mO2-T_c{T_c}-rho{rho}-D{D}-B{B}-n{n}-m{m}-f{'_'.join(str(fi) for fi in f)}-c{c}.pdf"
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

    cmp_Tfuncs_clock(T_c, rho, D, B, n, m_values, f, c,
                     m_values, r'$m$',
                     T_SA_APP_cont_mfc, r'$T_{SA\_APP}$', T_NCP_APP_cont_mfc, r'$T_{NCP\_APP}$',
                     yscale='log')

def SA_and_NCP_vs_n():
    # Constants (arbitrary but should be about reasonable scale)
    T_c = 1.0e-08 # on laptop
    rho = 2
    D   = 0.6
    B   = 0.24
    n_values = np.linspace(1, 60, 500)
    m   = 30
    f   = 1
    c   = 16

    cmp_Tfuncs_clock(T_c, rho, D, B, n_values, m, f, c,
                     n_values, r'$n$',
                     T_SA_APP_cont_mfc, r'$T_{SA\_APP}$', T_NCP_APP_cont_mfc, r'$T_{NCP\_APP}$',
                     yscale='log')

def NCP_and_CP_vs_m():
    T_c = 1.0e-08
    rho = 2
    D   = 0.6
    B   = 0.24
    n   = 40
    m_values = np.linspace(15, 35, 500)
    f   = 0.05
    c   = 16
    cmp_Tfuncs_clock(T_c, rho, D, B, n, m_values, f, c,
                     m_values, r'$m$',
                     T_NCP_APP_cont_mfc, r'$T_{NCP\_APP}$', T_CP_APP_cont_mfc, r'$T_{CP\_APP}$',
                    T_CP_APP_cont_mfc_litterature, r'$T_{CP\_APP\_lit}$',
                     yscale='log')


def NCP_and_CP_vs_n():
    T_c = 1.0e-08
    rho = 2
    D   = 0.6
    B   = 0.24
    n_values = np.linspace(15, 50, 500)
    m   = 30
    f   = 0.05
    c   = 16
    cmp_Tfuncs_clock(T_c, rho, D, B, n_values, m, f, c,
                     n_values, r'$n$',
                     T_NCP_APP_cont_mfc, r'$T_{NCP\_APP}$', T_CP_APP_cont_mfc, r'$T_{CP\_APP}$',
                     T_CP_APP_cont_mfc_litterature, r'$T_{CP\_APP\_lit}$',
                     yscale='log')


#SA_and_NCP_vs_m()
#SA_and_NCP_vs_n()
#NCP_vs_mO2()
NCP_and_CP_vs_m()
NCP_and_CP_vs_n()