"""
load_data.py
============
Load Kuczynski (1987) Table 4 data and compute fugacity coefficients
using the Peng-Robinson equation of state.

Thermodynamic constants taken directly from Kuczynski (1987) Table 5.
"""

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────
R = 8.3144          # J/(mol·K)
P0 = 0.1e6          # Standard state pressure = 0.1 MPa in Pa

# Peng-Robinson constants from Kuczynski Table 5
#                  CO        H2        CH3OH
Pc = np.array([3.45e6,   1.28e6,   8.078e6])   # Pa
Tc = np.array([134.0,    33.3,     512.6  ])   # K
omega = np.array([0.049, -0.226,    0.564  ])   # acentric factor
# Species index: 0=CO, 1=H2, 2=CH3OH

# Interaction parameters δ_ij assumed zero (as in Kuczynski)
delta = np.zeros((3, 3))

# Equilibrium constant correlation (Kuczynski Eq. 19)
def K_eq(T):
    """Thermodynamic equilibrium constant for CO + 2H2 = CH3OH"""
    return np.exp(-28.9762 + 11815.0 / T)


# ── Peng-Robinson EOS ──────────────────────────────────────────────────────
def pr_params(T):
    """Compute a_i(T) and b_i for each species."""
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    Tr = T / Tc
    alpha = (1 + kappa * (1 - np.sqrt(Tr)))**2
    a = 0.45724 * R**2 * Tc**2 / Pc * alpha   # N·m^4/mol^2
    b = 0.07780 * R * Tc / Pc                  # m^3/mol
    return a, b


def mixture_ab(y, T):
    """Mixture a and b using van der Waals mixing rules."""
    a_i, b_i = pr_params(T)
    # a_ij = (1 - delta_ij) * sqrt(a_i * a_j)
    a_ij = np.outer(np.sqrt(a_i), np.sqrt(a_i)) * (1 - delta)
    a_mix = np.sum(y[:, None] * y[None, :] * a_ij)
    b_mix = np.dot(y, b_i)
    return a_mix, b_mix, a_i, b_i, a_ij


def solve_Z(A, B):
    """
    Solve cubic Z^3 - (1-B)Z^2 + (A-3B^2-2B)Z - (AB-B^2-B^3) = 0
    Returns largest real root (vapour/gas phase).
    """
    coeffs = [1,
              -(1 - B),
              (A - 3*B**2 - 2*B),
              -(A*B - B**2 - B**3)]
    roots = np.roots(coeffs)
    real_roots = roots[np.isreal(roots)].real
    real_roots = real_roots[real_roots > B]   # physical constraint Z > B
    return float(np.max(real_roots))


def fugacity_coefficients(y, T, P):
    """
    Compute fugacity coefficients φ_i for a CO/H2/CH3OH mixture
    using Peng-Robinson EOS.

    Parameters
    ----------
    y : array [y_CO, y_H2, y_CH3OH]   mole fractions
    T : float   temperature (K)
    P : float   pressure (Pa)

    Returns
    -------
    phi : array [φ_CO, φ_H2, φ_CH3OH]
    """
    y = np.asarray(y, dtype=float)
    a_mix, b_mix, a_i, b_i, a_ij = mixture_ab(y, T)

    A = a_mix * P / (R * T)**2
    B = b_mix * P / (R * T)
    Z = solve_Z(A, B)

    # Kuczynski Eq. 17 for ln(φ_i)
    ln_phi = np.zeros(3)
    for i in range(3):
        sum_ya = 2 * np.sum(y * a_ij[i, :])
        ln_phi[i] = (b_i[i] / b_mix * (Z - 1)
                     - np.log(Z - B)
                     - A / (2 * np.sqrt(2) * B)
                     * (sum_ya / a_mix - b_i[i] / b_mix)
                     * np.log((Z + (1 + np.sqrt(2)) * B) /
                              (Z + (1 - np.sqrt(2)) * B)))
    return np.exp(ln_phi)


def reduced_fugacity(y, T, P):
    """
    Compute φ_i = P_T * y_i * f_i / P° (dimensionless fugacity)
    as used in Kuczynski rate expressions.

    Returns φ_CO, φ_H2, φ_CH3OH
    """
    phi = fugacity_coefficients(y, T, P)
    return P * np.array(y) * phi / P0


# ── Mole fractions from conversion ────────────────────────────────────────
def mole_fractions(zeta_CO, y_CO_in):
    """
    Compute mole fractions as function of CO conversion.
    Kuczynski Eqs. 14-16 (CO2-free feed).

    CO  + 2H2  →  CH3OH
    Feed: y_CO_in CO,  (1-y_CO_in) H2,  no CO2, no methanol
    """
    denom = 1 - 2 * zeta_CO * y_CO_in
    y_CO   = (1 - zeta_CO) * y_CO_in / denom
    y_H2   = (1 - y_CO_in - 2 * zeta_CO * y_CO_in) / denom
    y_MeOH = y_CO_in * zeta_CO / denom
    y = np.array([y_CO, y_H2, y_MeOH])
    # guard against negative (near-equilibrium)
    y = np.clip(y, 0.0, 1.0)
    y /= y.sum()
    return y


# ── Load and prepare the dataset ──────────────────────────────────────────
def load_data(filepath="kuczynski_data.csv"):
    """
    Load Table 4 and attach pre-computed inlet fugacities.

    Returns a DataFrame with columns:
        expt, P_MPa, P_Pa, y_CO_in, T_K, tau, zeta_CO,
        phi_CO_in, phi_H2_in, phi_MeOH_in
    """
    df = pd.read_csv(filepath)
    df["P_Pa"] = df["P_MPa"] * 1e6

    phi_CO_in   = []
    phi_H2_in   = []
    phi_MeOH_in = []

    for _, row in df.iterrows():
        y_in = np.array([row["y_CO_in"],
                         1.0 - row["y_CO_in"],
                         0.0])           # no methanol at inlet
        phi = reduced_fugacity(y_in, row["T_K"], row["P_Pa"])
        phi_CO_in.append(phi[0])
        phi_H2_in.append(phi[1])
        phi_MeOH_in.append(phi[2])

    df["phi_CO_in"]   = phi_CO_in
    df["phi_H2_in"]   = phi_H2_in
    df["phi_MeOH_in"] = phi_MeOH_in

    return df


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} experiments")
    print(df[["expt", "P_MPa", "y_CO_in", "T_K", "tau",
              "zeta_CO", "phi_CO_in", "phi_H2_in"]].head(10).to_string())
    print(f"\nT range:     {df.T_K.min():.1f} – {df.T_K.max():.1f} K")
    print(f"P range:     {df.P_MPa.min():.2f} – {df.P_MPa.max():.2f} MPa")
    print(f"y_CO range:  {df.y_CO_in.min():.3f} – {df.y_CO_in.max():.3f}")
    print(f"ζ_CO range:  {df.zeta_CO.min():.3f} – {df.zeta_CO.max():.3f}")
