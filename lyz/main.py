from functools import cache

import matplotlib
import numpy as np
import sympy
import sympy as smp
from matplotlib import pyplot as plt


d = 3
hbar, omega_0 = 1, 1  # hbar is 1.0545718e-34


@cache
def Z(N, beta):
    if N == 0:
        return 1
    elif N == 1:
        # The correct expression is equation (8) but in order to solve we use approximation (9)
        return 1 / ((beta * hbar * omega_0) ** d)
    return (1 / N) * sum([Z(1, k * beta) * Z(N - k, beta) for k in range(1, N + 1)])


def calculate_lyz(N):
    beta = smp.symbols(r'\beta', complex=True)
    Z_expr = Z(N, beta).subs({hbar: 1, omega_0: 1})
    Z_polynom = smp.Poly(Z_expr)
    lyz = np.roots(Z_polynom.all_coeffs())
    # We solve the roots for 1/beta.
    lyz = 1 / np.array(list(z for z in lyz if z))
    return lyz


def main():
    MIN_N = 2
    MAX_N = 20
    k_b = 1.380649e-23  # J*K^-1
    T_c = (hbar * omega_0 / k_b) * ((MAX_N / smp.zeta(d)) ** (1 / d))  # Equation (5). Where zeta(3) ~= 1.2
    beta_c = (1 / (k_b * T_c)).evalf()

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
        pass

    # TODO: Currently this is recreating Fig 1 panel (c)
    fig, ax = plt.subplots()
    scatter = ax.scatter(np.array(lyz_re) / beta_c, np.array(lyz_im) / beta_c,
                         c=color_scale, vmin=MIN_N, vmax=MAX_N, cmap='Blues')
    ax.scatter(1, 0, c='red')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('N')
    ax.set_xlabel(r'Re $\beta / \beta_c$')
    ax.set_ylabel(r'Im $\beta / \beta_c$')
    # ax.set_xlim([0.2, 1.5])
    # ax.set_ylim([-2, 2])
    ax.axhline(0, color='black', linewidth=1)
    plt.show()


if __name__ == '__main__':
    main()
