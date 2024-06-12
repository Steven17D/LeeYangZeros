import math
from functools import cache

import numpy as np
import sympy as smp
from matplotlib import pyplot as plt

hbar = 1  # hbar is 1.0545718e-34
k_b = 1  # 1.380649e-23 J*K^-1
omega_0 = 1


@cache
def P(N, y):
    """
    Equation (7) in the paper "Thermodynamic fermion-boson symmetry in harmonic oscillator potentials"
    https://arxiv.org/pdf/cond-mat/9810036
    """
    if N == 0 or N == 1:
        return 1

    return ((1 / smp.Integer(N)) * sum(
        ((math.prod((1 - y**j) ** 3 for j in range(N - n + 1, N + 1))) / ((1 - y**n) ** 3)) * P(N - n, y)
        for n in range(1, N + 1)
    )).simplify()


@cache
def Z(N, y):
    if N == 1:
        # Equation (4)
        return (y ** (smp.Integer(3) / 2)) / ((1 - y) ** 3)
    # Equation (6)
    return ((y ** (smp.Integer(N) * 3 / 2)) / math.prod((1 - y ** j) ** 3 for j in range(1, N + 1))) * P(N, y)


def calculate_lyz(N):
    y = smp.symbols('y')
    P_expr = P(N, y)
    P_poly = P_expr.simplify().as_poly()
    y_zeros = np.roots(P_poly.all_coeffs())
    lyz = np.log(y_zeros) / (- hbar * omega_0)
    return lyz


def calculate_T_c(N, d):
    """
    Calculate T_c according to equation (5) and equation (24).
    :param N: Number of particles
    :return: Critical temperature
    """
    if d == 1:
        # Equation (A4): (hbar * omega_0 / k_b) * (N / np.log(N))
        pass
    if d == 2:
        # Equation (5)
        return float((hbar * omega_0 / k_b) * ((N / smp.zeta(d)) ** (1 / d)))  # Equation (5)
    if d == 3:
        # Equation (24)
        T_c = smp.Symbol('T_c')
        T_c_solution, *_ = smp.solve(smp.Eq(T_c, (hbar * omega_0 / k_b) * ((N / smp.zeta(d)) ** (1 / 3)) * ((1 + ((3/2) * (smp.zeta(2) /smp.zeta(3)) * (hbar * omega_0 / (k_b * T_c)))) ** (-1/3))))
        return float(T_c_solution)
    raise ValueError("Only 3D and 2D are supported")


def main():
    d = 3
    MIN_N = 2
    MAX_N = 20

    lyz_re = []
    lyz_im = []
    color_scale = []
    try:
        for N in range(MIN_N, MAX_N + 1):
            lyz = calculate_lyz(N)
            print(f"N={N} Found roots({len(lyz)}): {lyz}")
            lyz_re += [smp.re(z_0) for z_0 in lyz]
            lyz_im += [smp.im(z_0) for z_0 in lyz]
            color_scale += [N] * len(lyz)
    except KeyboardInterrupt:
        N = N - 1

    beta_c = (1 / (k_b * calculate_T_c(N, d)))

    fig, ax = plt.subplots()
    scatter = ax.scatter(np.array(lyz_re) / beta_c, np.array(lyz_im) / beta_c,
                         c=color_scale, vmin=MIN_N, vmax=N, cmap='Blues')
    ax.scatter(1, 0, c='red')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('N')
    ax.set_xlabel(r'Re $\beta / \beta_c$')
    ax.set_ylabel(r'Im $\beta / \beta_c$')
    ax.set_xlim([0.3, 1.5])
    ax.set_ylim([-2, 2])
    ax.axhline(0, color='black', linewidth=1)
    plt.show()


if __name__ == '__main__':
    main()
