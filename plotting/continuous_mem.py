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

# Mem functions (T_) return: (core, all)

# cor:mem-cp-even
def M_CP_STA(M_z, n):
    core = 3 * 2 **(n/2 + 1) * M_z
    return (core, 0) #TODO: Implement if I need

def M_CP_STA_mfc(M_z, rho, B, n, m, f, c):
    (core, _) = M_CP_STA(M_z, n)
    if isinstance(m, list) or isinstance(m, np.ndarray):
        return ([core] * len(m), None)
    else:
        return (core, None)

# cor:mem-par-continuous
def M_PAR_STATES(M_z, rho, B, n, m_O23):
    core = (3* 2**(n/2) + rho ** (B * np.sqrt(n) * m_O23)) * M_z
    return (core, 0)

def M_PAR_STATES_mfc(M_z, rho, B, n, m, f, c):
    sqrt_n = np.sqrt(n)
    m_O1 = math.log(c/f,rho) / (B * sqrt_n)
    m_O23 = m - m_O1
    return M_PAR_STATES(M_z, rho, B, n, m_O23)


def sanitize_label(label):
    return re.sub(r'[^a-zA-Z0-9_-]', '', label)

# Comparing up to four functions agains some x_values
def cmp_Mfuncs_clock(M_z, rho, B, n, m, f, c, 
                     x_values, x_label, 
                     M_func1, M_func1_label, M_func2, M_func2_label, 
                     M_func3=None, M_func3_label=None, M_func4=None, M_func4_label=None,
                     yscale='linear'):
    (core_values1, _) = M_func1(M_z, rho, B, n, m, f, c)
    plt.plot(x_values, core_values1, label=M_func1_label, linestyle='--')
    (core_values2, _) = M_func2(M_z, rho, B, n, m, f, c)
    plt.plot(x_values, core_values2, label=M_func2_label, linestyle='-.')
    if (M_func3):
        (core_values3, _) = M_func3(M_z, rho, B, n, m, f, c)
        plt.plot(x_values, core_values3, label=M_func3_label, linestyle=':')
    if (M_func4):
        (core_values4, _) = M_func4(M_z, rho, B, n, m, f, c)
        plt.plot(x_values, core_values4, label=M_func4_label, linestyle='-')
    plt.xlabel(x_label)
    plt.ylabel(r'$M (byte)$')
    plt.yscale(yscale)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    filename = f"{sanitize_label(M_func1_label)}_and_{sanitize_label(M_func2_label)}"
    if M_func3:
        filename += f"_and_{sanitize_label(M_func3_label)}"
    if M_func4:
        filename += f"_and_{sanitize_label(M_func4_label)}"
    filename += f"-vs-{sanitize_label(x_label)}-M_z{M_z}-rho{rho}-B{B}"

    if not isinstance(n, np.ndarray):
        filename += f"-n{n}"
    if not isinstance(m, np.ndarray):
        filename += f"-m{m}"
    filename += f"-f{f}-c{c}.pdf"

    plt.savefig("./continuous_plots/" + filename)
    plt.show()

def CP_and_PAR_vs_n():
    # Constants (arbitrary but should be about reasonable scale)
    M_z = 8
    rho = 2
    B   = 0.24
    n_values = np.linspace(1, 60, 500)
    m   = 30
    f   = 0.05
    c   = 16
    cmp_Mfuncs_clock(M_z, rho, B, n_values, m, f, c,
                     n_values, r'$n$',
                     M_CP_STA_mfc, r'$M_{CP}$', M_PAR_STATES_mfc, r'$M_{PAR}$',
                     yscale='log')

def CP_and_PAR_vs_m():
    # Constants (arbitrary but should be about reasonable scale)
    M_z = 8
    rho = 2
    B   = 0.24
    n   = 40
    m_values   = np.linspace(1, 30, 500)
    f   = 0.05
    c   = 16
    cmp_Mfuncs_clock(M_z, rho, B, n, m_values, f, c,
                     m_values, r'$m$',
                     M_CP_STA_mfc, r'$M_{CP}$', M_PAR_STATES_mfc, r'$M_{PAR}$',
                     yscale='log')

CP_and_PAR_vs_n()
CP_and_PAR_vs_m()