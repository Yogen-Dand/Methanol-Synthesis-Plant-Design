"""
kinetics.py
===========
Shared kinetic module — import this in all downstream calculations.

Best-fit model: LH Model 1 (Kuczynski 1987, Eq. 20)
  Reaction:  CO + 2H2 = CH3OH
  Catalyst:  BASF S3-85  Cu/ZnO/Al2O3
  Conditions: T = 483–545 K,  P = 3–9 MPa,  CO-only synthesis gas

Usage
-----
    from kinetics import rate, K_eq
    r = rate(T=520, P_Pa=6e6, y={'CO': 0.10, 'H2': 0.90, 'MeOH': 0.0})
    # returns R_CO in mol CO / kg_cat / s
"""

import numpy as np
from load_data import reduced_fugacity, mole_fractions, K_eq

R_gas = 8.3144   # J/(mol·K)

# ─── Best-fit parameters (LH Model 1 / Kuczynski Eq. 20) ──────────────────
# k  = k0 · exp(-Ek/RT)       [mol CO / kg_cat / s]
# A  = A0 · exp(-EA/RT)        [dimensionless]
# B  = B0 · exp(-EB/RT)        [dimensionless]
# C  = C0                      [dimensionless]  ← found ≈ 0 by Kuczynski

# Values from Kuczynski Table 6  (used as defaults until your regression runs)
_PARAMS_LH1 = {
    "k0":  2.68e9,      # mol CO / kg_cat / s
    "Ek":  18400.0,     # K  (i.e. Ea/R)
    "A0":  0.069,
    "EA":  0.0,         # K
    "B0":  6.19e-8,
    "EB": -6610.0,      # K  (negative → B increases with T)
    "C0":  0.0,
}

# Alternative: LH Model 4 / Kuczynski Eq. 21
_PARAMS_LH4 = {
    "k0":  18.01,
    "Ek":  9032.0,
    "A0":  2.97e-4,
    "EA": -3539.0,
    "B0":  2.59e-4,
}


def _rate_LH1(T, phi_CO, phi_H2, phi_M, p=_PARAMS_LH1):
    """
    LH Model 1  (Kuczynski Eq. 20)
    r = k·(φ_CO·φ_H2² - φ_M/Keq) / (1 + A·φ_CO + B·φ_H2 + C·φ_M)³
    """
    k = p["k0"] * np.exp(-p["Ek"] / T)
    A = p["A0"] * np.exp(-p["EA"] / T)
    B = p["B0"] * np.exp(-p["EB"] / T)
    C = p["C0"]
    Keq = K_eq(T)
    driving = phi_CO * phi_H2**2 - phi_M / Keq
    denom   = (1 + A*phi_CO + B*phi_H2 + C*phi_M)**3
    return k * driving / denom


def _rate_LH4(T, phi_CO, phi_H2, phi_M, p=_PARAMS_LH4):
    """
    LH Model 4  (Kuczynski Eq. 21)
    r = k·(φ_CO·φ_H2² - φ_M/Keq) / (1 + A·φ_CO + B·φ_CO·φ_H2²)
    """
    k = p["k0"] * np.exp(-p["Ek"] / T)
    A = p["A0"] * np.exp(-p["EA"] / T)
    B = p["B0"]
    Keq = K_eq(T)
    driving = phi_CO * phi_H2**2 - phi_M / Keq
    denom   = 1 + A*phi_CO + B*phi_CO*phi_H2**2
    return k * driving / denom


def _rate_powerlaw(T, phi_CO, phi_H2, phi_M,
                   p={"ln_k0": 15.0, "Ea": 80000., "alpha": 1.0, "beta": 2.0}):
    """Power-law (no reversibility term) — for comparison only."""
    k = np.exp(p["ln_k0"]) * np.exp(-p["Ea"] / (R_gas * T))
    return k * (phi_CO**p["alpha"]) * (phi_H2**p["beta"])


# ─── Public API ───────────────────────────────────────────────────────────

def rate(T, P_Pa, y_dict, model="LH1", custom_params=None):
    """
    Compute the methanol synthesis rate for CO + 2H2 → CH3OH.

    Parameters
    ----------
    T         : float   Temperature (K)
    P_Pa      : float   Pressure (Pa), e.g. 6e6 for 6 MPa
    y_dict    : dict    Mole fractions, keys: 'CO', 'H2', 'MeOH'
                        (must sum to ~1; H2O not tracked in this CO2-free model)
    model     : str     'LH1'  — LH Model 1 / Eq.20  [recommended]
                        'LH4'  — LH Model 4 / Eq.21
                        'PL'   — Power law  (forward only)
    custom_params : dict  Override default parameters (optional)

    Returns
    -------
    R_CO : float   Reaction rate  [mol CO / kg_cat / s]
    """
    y_arr = np.array([y_dict.get("CO",   0.0),
                      y_dict.get("H2",   0.0),
                      y_dict.get("MeOH", 0.0)])
    y_arr = np.clip(y_arr, 0.0, 1.0)
    s = y_arr.sum()
    if s > 0:
        y_arr /= s

    phi = reduced_fugacity(y_arr, T, P_Pa)
    phi_CO, phi_H2, phi_M = phi[0], phi[1], phi[2]

    if model == "LH1":
        p = {**_PARAMS_LH1, **(custom_params or {})}
        return float(_rate_LH1(T, phi_CO, phi_H2, phi_M, p))
    elif model == "LH4":
        p = {**_PARAMS_LH4, **(custom_params or {})}
        return float(_rate_LH4(T, phi_CO, phi_H2, phi_M, p))
    elif model == "PL":
        p = {**{"ln_k0": 15.0, "Ea": 80000., "alpha": 1.0, "beta": 2.0},
             **(custom_params or {})}
        return float(_rate_powerlaw(T, phi_CO, phi_H2, phi_M, p))
    else:
        raise ValueError(f"Unknown model '{model}'. Use 'LH1', 'LH4', or 'PL'.")


def equilibrium_conversion(T, P_Pa, y_CO_in):
    """
    Estimate equilibrium CO conversion by finding ζ where driving force = 0.
    Uses bisection on  φ_CO·φ_H2² - φ_M/Keq = 0.
    """
    from scipy.optimize import brentq

    def driving_force(zeta):
        if zeta <= 0: return 1.0
        if zeta >= 0.999: return -1.0
        y = mole_fractions(zeta, y_CO_in)
        phi = reduced_fugacity(y, T, P_Pa)
        return phi[0] * phi[1]**2 - phi[2] / K_eq(T)

    try:
        return brentq(driving_force, 1e-4, 0.9999, xtol=1e-6)
    except ValueError:
        return np.nan


# ─── Quick self-test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    T   = 520.0         # K
    P   = 6.0e6         # Pa (6 MPa)
    y_in = {"CO": 0.102, "H2": 0.898, "MeOH": 0.0}

    for m in ["LH1", "LH4", "PL"]:
        r = rate(T, P, y_in, model=m)
        print(f"  {m:8s}  R_CO = {r:.4e}  mol/kg/s")

    zeq = equilibrium_conversion(T, P, y_CO_in=0.102)
    print(f"\n  Equilibrium conversion at {T} K, {P/1e6:.0f} MPa: ζ_eq = {zeq:.3f}")
