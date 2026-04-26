# benchmark.py
# Shared benchmark for CI Project — CSTR system from Cohen et al.
# Import this in your algorithm: from benchmark import get_cstr_experiments, GROUND_TRUTH

import numpy as np
from scipy.integrate import odeint

# ─────────────────────────────────────────────
# GROUND TRUTH (for reference only — not given to the algorithm)
# dcA/dt = 0.5*(cAf - cA) - cA^2
# Parameters: dilution rate D=0.5, rate constant k=1.0
# ─────────────────────────────────────────────
GROUND_TRUTH = "dcA/dt = 0.5*(cAf - cA) - cA^2"
GROUND_TRUTH_PARAMS = {"D": 0.5, "k": 1.0}

def _cstr_ode(cA, t, cA_f):
    return 0.5 * (cA_f - cA) - cA**2

def get_cstr_experiments(noise=0.05, seed=0):
    """
    Returns two CSTR experiments at different feed concentrations.
    Each experiment is a (t, cA_measured, var_funcs) tuple ready
    for use in Cohen's fitness function.

    noise : fractional noise level (0.05 = 5%)
    seed  : random seed for reproducibility

    Usage:
        from benchmark import get_cstr_experiments
        experiments = get_cstr_experiments(noise=0.05, seed=0)
    """
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, 4, 50)

    experiments = []
    for cA_f in [1.0, 0.5]:
        clean = odeint(_cstr_ode, 0, t, args=(cA_f,)).flatten()
        scale = noise * np.mean(clean)
        noisy = clean + rng.normal(scale=scale, size=clean.shape)

        cA_f_val = cA_f  # capture for lambda
        var_funcs = {
            "c_A":   lambda cA: cA,
            "c_Af":  lambda cA, v=cA_f_val: v,
            "c_A^2": lambda cA: cA**2,
        }
        experiments.append((t, noisy, var_funcs))

    return experiments

def get_cstr_arrays(noise=0.05, seed=0):
    """
    Same data but as flat numpy arrays — for algorithms that
    don't use the ODE integration approach (e.g. algebraic SR).

    Returns:
        X : array of shape (N, 3) — columns: [cA, cAf, cA^2]
        y : array of shape (N,)   — measured dcA/dt (finite differences)
    """
    experiments = get_cstr_experiments(noise=noise, seed=seed)
    X_list, y_list = [], []

    for (t, cA_meas, _) in experiments:
        # Estimate dcA/dt from finite differences
        dcA_dt = np.gradient(cA_meas, t)
        cA_f   = 1.0 if cA_meas.mean() > 0.4 else 0.5  # infer from data
        X_list.append(np.column_stack([
            cA_meas,
            np.full_like(cA_meas, cA_f),
            cA_meas**2
        ]))
        y_list.append(dcA_dt)

    return np.vstack(X_list), np.concatenate(y_list)


if __name__ == "__main__":
    # Quick sanity check — run this file directly to verify data looks right
    import matplotlib.pyplot as plt

    experiments = get_cstr_experiments(noise=0.05, seed=0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, (t, cA_meas, var_funcs) in enumerate(experiments):
        cA_f = var_funcs["c_Af"](0)
        # clean reference
        clean = odeint(_cstr_ode, 0, t, args=(cA_f,)).flatten()
        axes[i].plot(t, clean,   label="Clean",   linewidth=2)
        axes[i].plot(t, cA_meas, label="Noisy",   linestyle="--", alpha=0.7)
        axes[i].set_title(f"Experiment {i+1}: cAf={cA_f}")
        axes[i].set_xlabel("t")
        axes[i].set_ylabel("cA")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.suptitle(f"CSTR Benchmark Data\n{GROUND_TRUTH}")
    plt.tight_layout()
    plt.savefig("cstr_benchmark.png", dpi=150)
    plt.show()
    print(f"Ground truth: {GROUND_TRUTH}")
    print(f"Experiments: {len(experiments)} conditions")
    print(f"Timesteps per experiment: {len(experiments[0][0])}")