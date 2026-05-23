"""
fit_models.py
=============
Fit three rival kinetic models to Kuczynski (1987) integral reactor data.

Strategy
--------
Because the reactor is integral (not differential), we cannot fit rates
directly. Instead, for each trial parameter set we:
  1. Integrate  dζ/dτ = R_CO(T, P, ζ)  from τ=0
  2. Read off ζ_predicted at the experimental τ
  3. Minimise Σ(ζ_pred - ζ_obs)²

Three models
------------
A  Power Law           r = k0·exp(-Ea/RT)·φ_CO^α · φ_H2^β
B  LH Model 1 (Eq.20)  r = k(φ_CO·φ_H2² - φ_M/Keq) / (1 + A·φ_CO + B·φ_H2 + C·φ_M)³
C  LH Model 4 (Eq.21)  r = k(φ_CO·φ_H2² - φ_M/Keq) / (1 + A·φ_CO + B·φ_CO·φ_H2²)
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, least_squares
import warnings
warnings.filterwarnings("ignore")

from load_data import load_data, mole_fractions, reduced_fugacity, K_eq

R_gas = 8.3144   # J/(mol·K)

# ─────────────────────────────────────────────────────────────────────────────
# Rate expressions (all return R_CO in mol CO / kg_cat / s)
# φ values are dimensionless reduced fugacities (= P·y·f / P°)
# ─────────────────────────────────────────────────────────────────────────────

def rate_A(T, phi_CO, phi_H2, phi_M, params):
    """Power law: r = k0·exp(-Ea/RT) · φ_CO^α · φ_H2^β  (no reversibility)"""
    ln_k0, Ea, alpha, beta = params
    k = np.exp(ln_k0) * np.exp(-Ea / (R_gas * T))
    return k * (phi_CO ** alpha) * (phi_H2 ** beta)


def rate_B(T, phi_CO, phi_H2, phi_M, params):
    """
    LH Model 1 (Kuczynski Eq. 20)
    r = k·(φ_CO·φ_H2² - φ_M/Keq) / (1 + A·φ_CO + B·φ_H2 + C·φ_M)³
    params: [ln_k0, Ek, ln_A0, EA, ln_B0, EB, ln_C0]
    """
    ln_k0, Ek, ln_A0, EA, ln_B0, EB, ln_C0 = params
    k  = np.exp(ln_k0) * np.exp(-Ek  / (R_gas * T))
    A  = np.exp(ln_A0) * np.exp(-EA  / (R_gas * T))
    B  = np.exp(ln_B0) * np.exp(-EB  / (R_gas * T))
    C  = np.exp(ln_C0)                              # Kuczynski found C0 = 0

    Keq = K_eq(T)
    driving = phi_CO * phi_H2**2 - phi_M / Keq
    denom = (1 + A * phi_CO + B * phi_H2 + C * phi_M) ** 3
    return k * driving / denom


def rate_C(T, phi_CO, phi_H2, phi_M, params):
    """
    LH Model 4 (Kuczynski Eq. 21)
    r = k·(φ_CO·φ_H2² - φ_M/Keq) / (1 + A·φ_CO + B·φ_CO·φ_H2²)
    params: [ln_k0, Ek, ln_A0, EA, ln_B0]
    """
    ln_k0, Ek, ln_A0, EA, ln_B0 = params
    k = np.exp(ln_k0) * np.exp(-Ek / (R_gas * T))
    A = np.exp(ln_A0) * np.exp(-EA / (R_gas * T))
    B = np.exp(ln_B0)

    Keq = K_eq(T)
    driving = phi_CO * phi_H2**2 - phi_M / Keq
    denom = 1 + A * phi_CO + B * phi_CO * phi_H2**2
    return k * driving / denom


MODELS = {
    "A_PowerLaw": rate_A,
    "B_LH_Model1": rate_B,
    "C_LH_Model4": rate_C,
}

# ─────────────────────────────────────────────────────────────────────────────
# Integral reactor: integrate dζ/dτ = R_CO and read off ζ at τ_exp
# ─────────────────────────────────────────────────────────────────────────────

def predict_conversion(T, P_Pa, y_CO_in, tau_values, params, rate_fn):
    """
    Integrate the plug-flow mole balance for one experimental group.
    Returns array of predicted ζ_CO at each τ in tau_values.
    """
    tau_max = float(np.max(tau_values)) * 1.05

    def dzeta_dtau(tau, zeta):
        z = float(np.clip(zeta[0], 0.0, 0.999))
        y = mole_fractions(z, y_CO_in)
        phi = reduced_fugacity(y, T, P_Pa)
        r = rate_fn(T, phi[0], phi[1], phi[2], params)
        r = max(r, 0.0)    # rate cannot be negative (forward direction only here)
        return [r]

    sol = solve_ivp(
        dzeta_dtau,
        [0.0, tau_max],
        [0.0],
        method="RK45",
        t_eval=np.sort(tau_values),
        rtol=1e-5,
        atol=1e-7,
        max_step=tau_max / 50,
    )

    # interpolate to exact tau_values requested
    zeta_pred = np.interp(tau_values, sol.t, sol.y[0])
    return np.clip(zeta_pred, 0.0, 1.0)


def residuals(params, df, rate_fn):
    """Compute residual vector (ζ_pred - ζ_obs) for all 110 experiments."""
    res = []
    # Group by (T, P, y_CO_in) so we integrate once per group
    for (T, P_Pa, y_CO_in), grp in df.groupby(["T_K", "P_Pa", "y_CO_in"],
                                                sort=False):
        tau_vals = grp["tau"].values
        zeta_obs = grp["zeta_CO"].values
        try:
            zeta_pred = predict_conversion(T, P_Pa, y_CO_in, tau_vals,
                                           params, rate_fn)
            res.extend(zeta_pred - zeta_obs)
        except Exception:
            res.extend([1.0] * len(tau_vals))   # penalise failed integration
    return np.array(res)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter bounds (log-space where appropriate for positivity)
# ─────────────────────────────────────────────────────────────────────────────

BOUNDS = {
    "A_PowerLaw": [
        # ln_k0,  Ea (J/mol),  alpha,  beta
        [(-5, 25), (5000, 200000), (0.1, 3.0), (0.5, 4.0)],
    ],
    "B_LH_Model1": [
        # ln_k0, Ek,       ln_A0,  EA,         ln_B0,  EB,       ln_C0
        [(-5, 25), (5000, 200000),
         (-10, 5), (-80000, 80000),
         (-25, 5), (-80000, 80000),
         (-10, 5)],
    ],
    "C_LH_Model4": [
        # ln_k0, Ek,       ln_A0,  EA,         ln_B0
        [(-5, 25), (5000, 200000),
         (-15, 5), (-80000, 80000),
         (-25, 5)],
    ],
}


def fit_model(df, model_name, n_restarts=8, seed=42):
    """
    Fit a model using differential evolution (global) followed by
    Levenberg-Marquardt local refinement.
    Returns best_params, SSR, residuals_vector
    """
    rate_fn = MODELS[model_name]
    bounds  = BOUNDS[model_name][0]

    print(f"\n{'='*55}")
    print(f"  Fitting {model_name}  ({len(bounds)} params, {n_restarts} DE restarts)")
    print(f"{'='*55}")

    def objective(params):
        r = residuals(params, df, rate_fn)
        return float(np.sum(r**2))

    # Global search with differential evolution
    np.random.seed(seed)
    result_global = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=300,
        tol=1e-6,
        seed=seed,
        workers=1,
        popsize=10,
        mutation=(0.5, 1.5),
        recombination=0.7,
        disp=False,
    )
    print(f"  DE best SSR: {result_global.fun:.6f}")

    # Local refinement
    result_local = least_squares(
        residuals,
        result_global.x,
        args=(df, rate_fn),
        method="lm",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=5000,
    )
    final_params = result_local.x
    res_vec      = result_local.fun
    SSR          = float(np.sum(res_vec**2))
    print(f"  Final SSR:   {SSR:.6f}")

    return final_params, SSR, res_vec


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data("kuczynski_data.csv")
    print(f"Dataset: {len(df)} experiments loaded\n")

    results = {}
    for model_name in MODELS:
        params, ssr, res = fit_model(df, model_name)
        results[model_name] = {"params": params, "SSR": ssr, "residuals": res}

    print("\n\n" + "="*55)
    print("  RESULTS SUMMARY")
    print("="*55)
    n = len(df)
    for name, r in results.items():
        k  = len(r["params"])
        ssr = r["SSR"]
        aic = 2 * k + n * np.log(ssr / n)
        rmse = np.sqrt(ssr / n)
        print(f"\n{name}")
        print(f"  Params : {k}")
        print(f"  SSR    : {ssr:.5f}")
        print(f"  RMSE   : {rmse:.4f}")
        print(f"  AIC    : {aic:.2f}")
        print(f"  Params : {np.round(r['params'], 4).tolist()}")

    # Save results
    np.save("fit_results.npy", results, allow_pickle=True)
    print("\nResults saved to fit_results.npy")
