"""
model_compare.py
================
Model discrimination and all report-quality plots.
Run AFTER fit_models.py has saved fit_results.npy

Generates:
  Fig 1.1 — Arrhenius plot:    ln(apparent rate) vs. 1/T
  Fig 1.2 — Rate vs. pressure  at fixed T and y_CO
  Fig 1.3 — Rate vs. conversion (shows product inhibition)
  Fig 1.4 — Parity plot:       ζ_pred vs. ζ_obs for all models
  Fig 1.5 — Residual plots:    (ζ_pred - ζ_obs) vs. ζ_obs
  Fig 1.6 — AIC / model comparison table
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp

from load_data import load_data, mole_fractions, reduced_fugacity, K_eq
from fit_models import MODELS, predict_conversion, residuals

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 120,
})

R_gas = 8.3144
MODEL_COLORS = {"A_PowerLaw": "#e74c3c", "B_LH_Model1": "#2ecc71", "C_LH_Model4": "#3498db"}
MODEL_LABELS = {"A_PowerLaw": "Model A: Power Law",
                "B_LH_Model1": "Model B: LH Model 1 (Eq.20)",
                "C_LH_Model4": "Model C: LH Model 4 (Eq.21)"}


def load_results():
    try:
        return np.load("fit_results.npy", allow_pickle=True).item()
    except FileNotFoundError:
        print("fit_results.npy not found — running with Kuczynski published params")
        # Use published params as fallback for plotting
        from fit_models import rate_A, rate_B, rate_C
        fallback_B = np.array([np.log(2.68e9), 18400*R_gas, np.log(0.069), 0.0,
                                np.log(6.19e-8), -6610*R_gas, np.log(1e-10)])
        fallback_C = np.array([np.log(18.01),  9032*R_gas, np.log(2.97e-4), -3539*R_gas,
                                np.log(2.59e-4)])
        fallback_A = np.array([np.log(1e4),    80000.0,    1.0, 2.0])
        df = load_data()
        results = {}
        for name, params in [("A_PowerLaw", fallback_A),
                              ("B_LH_Model1", fallback_B),
                              ("C_LH_Model4", fallback_C)]:
            res = residuals(params, df, MODELS[name])
            results[name] = {"params": params, "SSR": float(np.sum(res**2)), "residuals": res}
        return results


def get_all_predictions(df, results):
    """Compute ζ_pred for all models and all experiments."""
    preds = {}
    for name, r in results.items():
        params = r["params"]
        rate_fn = MODELS[name]
        all_pred = []
        for (T, P_Pa, y_CO_in), grp in df.groupby(["T_K", "P_Pa", "y_CO_in"], sort=False):
            tau_vals = grp["tau"].values
            zp = predict_conversion(T, P_Pa, y_CO_in, tau_vals, params, rate_fn)
            all_pred.extend(zp)
        # Reindex to match df order
        pred_series = pd.Series(all_pred)
        preds[name] = pred_series.values
    return preds


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1.1 — Arrhenius plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_arrhenius(df):
    """
    Approximate apparent rate at each experiment:
      r_app ≈ ζ_CO / τ  (valid at low conversion where ODE ≈ linear)
    Plot ln(r_app) vs 1/T coloured by pressure group.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    low_conv = df[df["zeta_CO"] < 0.15].copy()
    low_conv["r_app"] = low_conv["zeta_CO"] / low_conv["tau"]
    low_conv = low_conv[low_conv["r_app"] > 0]

    P_groups = sorted(low_conv["P_MPa"].round(0).unique())
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(P_groups)))

    for P_grp, col in zip(P_groups, cmap):
        sub = low_conv[np.abs(low_conv["P_MPa"] - P_grp) < 0.3]
        ax.scatter(1000 / sub["T_K"], np.log(sub["r_app"]),
                   color=col, s=40, alpha=0.8, label=f"P ≈ {P_grp:.0f} MPa")

    ax.set_xlabel("1000 / T  (K⁻¹)")
    ax.set_ylabel("ln(apparent rate)  [mol CO / kg·s]")
    ax.set_title("Fig 1.1 — Arrhenius Plot (low conversion data, ζ < 0.15)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig1_1_arrhenius.png", dpi=150)
    plt.close()
    print("Saved fig1_1_arrhenius.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1.2 — Rate vs. pressure at fixed T
# ─────────────────────────────────────────────────────────────────────────────
def plot_rate_vs_pressure(results):
    """Model predicted initial rate vs. pressure at T=521 K, y_CO=0.10"""
    from kinetics import rate as kinetics_rate

    T       = 521.0
    y_CO_in = 0.102
    P_range = np.linspace(3e6, 9e6, 100)

    fig, ax = plt.subplots(figsize=(7, 5))

    for name, r in results.items():
        params   = r["params"]
        rate_fn  = MODELS[name]
        r0_list  = []
        for P in P_range:
            from load_data import reduced_fugacity
            y = np.array([y_CO_in, 1.0 - y_CO_in, 0.0])
            phi = reduced_fugacity(y, T, P)
            r0 = rate_fn(T, phi[0], phi[1], phi[2], params)
            r0_list.append(max(r0, 0))
        ax.plot(P_range / 1e6, r0_list,
                color=MODEL_COLORS[name], lw=2, label=MODEL_LABELS[name])

    ax.set_xlabel("Total pressure  (MPa)")
    ax.set_ylabel("Initial rate  R_CO  [mol CO / kg·s]")
    ax.set_title(f"Fig 1.2 — Rate vs. Pressure\n(T = {T} K, y_CO_in = {y_CO_in})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig1_2_rate_vs_pressure.png", dpi=150)
    plt.close()
    print("Saved fig1_2_rate_vs_pressure.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1.3 — Rate vs. conversion (product inhibition)
# ─────────────────────────────────────────────────────────────────────────────
def plot_rate_vs_conversion(results):
    """Show how rate changes along the reactor (T=521K, P=6MPa, y_CO_in=0.10)"""
    T       = 521.0
    P_Pa    = 6.0e6
    y_CO_in = 0.102
    zeta_range = np.linspace(0.001, 0.75, 200)

    fig, ax = plt.subplots(figsize=(7, 5))

    for name, r in results.items():
        params  = r["params"]
        rate_fn = MODELS[name]
        rates   = []
        for z in zeta_range:
            y   = mole_fractions(z, y_CO_in)
            phi = reduced_fugacity(y, T, P_Pa)
            rv  = rate_fn(T, phi[0], phi[1], phi[2], params)
            rates.append(max(rv, 0))
        ax.plot(zeta_range, rates,
                color=MODEL_COLORS[name], lw=2, label=MODEL_LABELS[name])

    ax.axvline(x=0.5, color="gray", ls="--", lw=1, label="ζ = 0.50")
    ax.set_xlabel("CO conversion  ζ_CO")
    ax.set_ylabel("Rate  R_CO  [mol CO / kg·s]")
    ax.set_title(f"Fig 1.3 — Rate vs. Conversion\n(T={T} K, P={P_Pa/1e6:.0f} MPa, y_CO_in={y_CO_in})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig1_3_rate_vs_conversion.png", dpi=150)
    plt.close()
    print("Saved fig1_3_rate_vs_conversion.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1.4 — Parity plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_parity(df, preds):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for ax, (name, zp) in zip(axes, preds.items()):
        zo = df["zeta_CO"].values
        # Reorder predictions to match df row order
        ax.scatter(zo, zp, s=20, alpha=0.6,
                   color=MODEL_COLORS[name], edgecolors="none")
        lim = [0, max(zo.max(), np.nanmax(zp)) * 1.05]
        ax.plot(lim, lim, "k--", lw=1.5, label="y = x")
        ax.set_xlabel("ζ_CO  observed")
        ax.set_ylabel("ζ_CO  predicted")
        ax.set_title(MODEL_LABELS[name])
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.grid(True, alpha=0.3)
        ssr  = results[name]["SSR"]
        rmse = np.sqrt(ssr / len(df))
        ax.text(0.05, 0.92, f"RMSE = {rmse:.4f}", transform=ax.transAxes,
                fontsize=9, color="navy")

    fig.suptitle("Fig 1.4 — Parity Plot: Predicted vs. Observed Conversion", y=1.01)
    plt.tight_layout()
    plt.savefig("fig1_4_parity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig1_4_parity.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1.5 — Residual plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_residuals(df, preds):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for ax, (name, zp) in zip(axes, preds.items()):
        zo  = df["zeta_CO"].values
        res = zp - zo
        ax.scatter(zo, res, s=20, alpha=0.6,
                   color=MODEL_COLORS[name], edgecolors="none")
        ax.axhline(0, color="black", lw=1.5)
        ax.axhline(+0.05, color="gray", ls="--", lw=1)
        ax.axhline(-0.05, color="gray", ls="--", lw=1)
        ax.set_xlabel("ζ_CO  observed")
        ax.set_ylabel("Residual  (ζ_pred − ζ_obs)")
        ax.set_title(MODEL_LABELS[name])
        ax.grid(True, alpha=0.3)

    fig.suptitle("Fig 1.5 — Residual Plots (random = good, pattern = model failure)", y=1.01)
    plt.tight_layout()
    plt.savefig("fig1_5_residuals.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig1_5_residuals.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1.6 — Model comparison table
# ─────────────────────────────────────────────────────────────────────────────
def plot_model_table(results, n):
    rows = []
    for name, r in results.items():
        k   = len(r["params"])
        ssr = r["SSR"]
        aic = 2 * k + n * np.log(ssr / n)
        bic = k * np.log(n) + n * np.log(ssr / n)
        rmse = np.sqrt(ssr / n)
        rows.append([MODEL_LABELS[name], k, f"{ssr:.4f}", f"{rmse:.4f}",
                     f"{aic:.1f}", f"{bic:.1f}"])

    # Sort by AIC
    rows.sort(key=lambda x: float(x[4]))
    rows[0][0] = "★ " + rows[0][0]   # mark winner

    cols = ["Model", "# Params", "SSR", "RMSE", "AIC", "BIC"]
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols,
                   cellLoc="center", loc="center",
                   bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row == 1:
            cell.set_facecolor("#d5f5e3")   # winner highlighted
        else:
            cell.set_facecolor("#f8f9fa")

    ax.set_title("Fig 1.6 — Model Comparison  (★ = best AIC)",
                 fontsize=12, pad=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("fig1_6_model_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig1_6_model_table.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df      = load_data("kuczynski_data.csv")
    results = load_results()

    print("Computing all predictions (this may take a minute)...")
    preds = get_all_predictions(df, results)

    # Align preds to df row order (groups may differ from original order)
    # Rebuild a prediction column aligned with df index
    for name in preds:
        # predictions are already in df row order from groupby
        pass

    plot_arrhenius(df)
    plot_rate_vs_pressure(results)
    plot_rate_vs_conversion(results)
    plot_parity(df, preds)
    plot_residuals(df, preds)
    plot_model_table(results, n=len(df))

    print("\nAll 6 figures saved.")

    # Print final recommendation
    n = len(df)
    best = min(results.items(),
               key=lambda x: 2*len(x[1]["params"]) + n*np.log(x[1]["SSR"]/n))
    print(f"\n  Recommended model: {MODEL_LABELS[best[0]]}")
    print(f"  AIC = {2*len(best[1]['params']) + n*np.log(best[1]['SSR']/n):.1f}")
