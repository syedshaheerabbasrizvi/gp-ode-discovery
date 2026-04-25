import numpy as np
import operator
import random
import warnings
import logging
import math
from scipy.optimize import curve_fit, OptimizeWarning
from deap import base, creator, tools, gp, algorithms
from tqdm import tqdm
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    filename="gp_feynman_I6_2a.log",
    filemode="w",
    format="%(message)s",
    level=logging.INFO
)

# ─────────────────────────────────────────────
# SECTION 1: PARAM INCLUSION (unchanged from original)
# ─────────────────────────────────────────────

def param_incl(exp_tree, unary_prim, bin_prim, arg_set, n_start):
    T = {}
    for arg in arg_set:
        T[arg] = lambda vars, thetas, arg=arg: vars[arg]

    if len(exp_tree) > 1:
        k = 0
        n = n_start + 1

        while len(exp_tree) > 1:
            if (exp_tree[k] in unary_prim) and (exp_tree[k+1] in T):
                L   = T[exp_tree[k+1]]
                op  = unary_prim[exp_tree[k]]
                key = exp_tree[k] + exp_tree[k+1]
                Tn  = lambda vars, thetas, L=L, op=op, n=n: thetas[n] * op(L(vars, thetas))
                T[key] = Tn
                del exp_tree[k+1]
                exp_tree[k] = key
                n += 1
                k = 0

            elif (exp_tree[k] in bin_prim) and (exp_tree[k+1] in T) and (exp_tree[k+2] in T):
                L   = T[exp_tree[k+1]]
                R   = T[exp_tree[k+2]]
                op  = bin_prim[exp_tree[k]]
                key = exp_tree[k+1] + exp_tree[k] + exp_tree[k+2]

                if exp_tree[k] in {"+", "-"}:
                    Tn = lambda vars, thetas, L=L, R=R, op=op, n=n: op(L(vars, thetas), thetas[n] * R(vars, thetas))
                    n += 1
                else:
                    Tn = lambda vars, thetas, L=L, R=R, op=op: op(L(vars, thetas), R(vars, thetas))

                T[key] = Tn
                del exp_tree[k+2]
                del exp_tree[k+1]
                exp_tree[k] = key
                k = 0
            else:
                k += 1

    final    = T[exp_tree[0]]
    n_params = n - n_start
    f = lambda vars, thetas, final=final: thetas[n_start] * final(vars, thetas)
    return f, n_params


# ─────────────────────────────────────────────
# SECTION 2: FITNESS — DIRECT FUNCTION REGRESSION
# (No ODE; just evaluate f(vars, thetas) per data row)
# ─────────────────────────────────────────────

def safe_exp(x):
    return np.exp(np.clip(x, -100, 100))

def safe_div(a, b):
    return np.where(np.abs(b) < 1e-10, 1e10, a / b)

def safe_sq(x):
    return x * x

def fitness(individual, data_arrays, y_measured, gp_config, n_start=0):
    try:
        f, n_params = param_incl(
            list(individual),
            gp_config["unary_prim"],
            gp_config["bin_prim"],
            gp_config["arg_set"],
            n_start
        )

        def model(dummy_X, *thetas):
            try:
                vals = f(data_arrays, thetas)  # entire array at once
                if not np.all(np.isfinite(vals)):
                    return np.full(len(y_measured), 1e10)
                return vals
            except Exception:
                return np.full(len(y_measured), 1e10)

        n_data  = len(y_measured)
        dummy_X = np.zeros(n_data)
        p0      = [1.0] * n_params

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            popt, _ = curve_fit(model, dummy_X, y_measured, p0=p0, maxfev=5000)

        predicted = model(dummy_X, *popt)
        MSE = np.mean((predicted - y_measured) ** 2)
        n_nodes = len(individual)  # total node count, not just params
        BIC = n_data * np.log(MSE + 1e-30) + n_params * np.log(n_data) * 3
        return (BIC,)

    except Exception:
        return (1e10,)


# ─────────────────────────────────────────────
# SECTION 3: DEAP SETUP — extended primitive set
# Target: exp(-theta^2 / (2*sigma^2)) / sqrt(2*pi*sigma^2)
# Needed primitives: exp, sq (x^2), div, mul, add, sub
# Terminals: theta, sigma, theta^2, sigma^2
# ─────────────────────────────────────────────

pset = gp.PrimitiveSet("f", arity=0)
pset.addPrimitive(operator.add,  2, name="+")
pset.addPrimitive(operator.sub,  2, name="-")
pset.addPrimitive(operator.mul,  2, name="*")
pset.addPrimitive(safe_div,      2, name="/")
pset.addPrimitive(safe_exp,      1, name="exp")
pset.addPrimitive(safe_sq,       1, name="sq")

# Terminals: raw variables and useful precomputed combos
pset.addTerminal("theta",   name="theta")
pset.addTerminal("sigma",   name="sigma")
pset.addTerminal("theta^2", name="theta^2")
pset.addTerminal("sigma^2", name="sigma^2")

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr",       gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate",       gp.cxOnePoint)
toolbox.register("mutate",     gp.mutUniform, expr=toolbox.expr, pset=pset)
toolbox.register("select",     tools.selTournament, tournsize=2)
toolbox.decorate("mate",   gp.staticLimit(key=operator.attrgetter("height"), max_value=6))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6))
toolbox.decorate("mate",   gp.staticLimit(key=len, max_value=20))
toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=20))

# ─────────────────────────────────────────────
# SECTION 4: WIRE DATA AND FITNESS
# ─────────────────────────────────────────────

bin_prim_funcs   = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": safe_div}
unary_prim_funcs = {"exp": safe_exp, "sq": safe_sq}
arg_set = {"theta", "sigma", "theta^2", "sigma^2"}

gp_config = {
    "unary_prim": unary_prim_funcs,
    "bin_prim":   bin_prim_funcs,
    "arg_set":    arg_set
}

def load_feynman_dataset(filepath, n_samples=None, noise_frac=0.0, seed=0):
    """
    Load a Feynman dataset file.
    Format: space/tab delimited, columns = [var1, var2, ..., y]
    For I.6.2a: columns are [theta, sigma, y]

    n_samples   : if set, randomly subsample this many rows (dataset has 10^6 rows)
    noise_frac  : add Gaussian noise as fraction of y std (0.0 = use clean data as-is)
    """
    raw = np.loadtxt(filepath)  # shape (N, D+1)
    
    if n_samples is not None and n_samples < len(raw):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(raw), size=n_samples, replace=False)
        raw = raw[idx]
    
    X = raw[:, :-1]   # all columns except last → input variables
    y = raw[:, -1]    # last column → target
    
    if noise_frac > 0.0:
        rng = np.random.default_rng(seed)
        y = y + rng.normal(scale=noise_frac * np.std(y), size=len(y))
    
    return X, y


# ── Load the actual Feynman file ──
# Variables for I.6.2a: col 0 = theta, col 1 = sigma, col 2 = y
X, y = load_feynman_dataset(
    filepath="feynman_with_units/I.6.2",
    n_samples=500,
    noise_frac=0.01,
    seed=0
)


theta_vals = X[:, 0]
sigma_vals = X[:, 1]

data_arrays = {
    "theta":   theta_vals,
    "sigma":   sigma_vals,
    "theta^2": theta_vals**2,
    "sigma^2": sigma_vals**2,
}
y_measured_global = y

def evaluate_individual(ind):
    return fitness([node.name for node in ind], data_arrays, y_measured_global, gp_config)

toolbox.register("evaluate", evaluate_individual)

# ─────────────────────────────────────────────
# SECTION 5: MAIN GP LOOP (unchanged structure)
# ─────────────────────────────────────────────

def run_gp(n_pop=200, n_gen=30, n_hof=10, cx_prob=0.6, mut_prob=0.2, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    pop = toolbox.population(n=n_pop)
    hof = tools.HallOfFame(n_hof)
    history_best      = []
    history_avg       = []
    history_worst     = []
    history_diversity = []

    with Pool() as pool:
        toolbox.register("map", pool.map)

        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        hof.update(pop)
        valid_fits = [ind.fitness.values[0] for ind in pop if ind.fitness.values[0] < 1e9]
        history_best.append(min(valid_fits))
        history_avg.append(np.mean(valid_fits))
        history_worst.append(max(valid_fits))
        history_diversity.append(len(set(str(ind) for ind in pop)) / n_pop)
        logging.info("Gen   0 | initial population evaluated")

        for gen in tqdm(range(n_gen), desc="Evolving"):
            offspring = algorithms.varAnd(pop, toolbox, cxpb=cx_prob, mutpb=mut_prob)
            n_immigrants = n_pop // 5
            immigrants   = toolbox.population(n=n_immigrants)
            fitnesses    = toolbox.map(toolbox.evaluate, immigrants)
            for ind, fit in zip(immigrants, fitnesses):
                ind.fitness.values = fit
            offspring = offspring + immigrants
            invalid   = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid)
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

            pop = toolbox.select(offspring + pop, k=n_pop)
            hof.update(pop)
            valid_fits = [ind.fitness.values[0] for ind in pop if ind.fitness.values[0] < 1e9]
            history_best.append(min(valid_fits))
            history_avg.append(np.mean(valid_fits))
            history_worst.append(max(valid_fits))
            history_diversity.append(len(set(str(ind) for ind in pop)) / n_pop)

            best = min(ind.fitness.values[0] for ind in pop)
            logging.info(f"Gen {gen+1:3d} | best BIC: {best:.3f}")

    return hof, {
        "best":      history_best,
        "avg":       history_avg,
        "worst":     history_worst,
        "diversity": history_diversity,
    }


# ─────────────────────────────────────────────
# SECTION 6: RUN AND RESULTS
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.info("Running GP on Feynman I.6.2a ...")
    hof, stats = run_gp(n_pop=500, n_gen=50, n_hof=10, cx_prob = 0.5, mut_prob=0.4)

    print("\n── Hall of Fame ──")
    for i, ind in enumerate(hof):
        print(f"{i+1:2d}. BIC={ind.fitness.values[0]:.3f}  expr={str(ind)}")

    # ── Fit and display parameters for all HoF individuals ──
    print("\n── Hall of Fame Parameter Fits ──")
    print(f"Ground truth: theta[0]=1/sqrt(2π)≈{1/np.sqrt(2*np.pi):.6f}, theta_inner=-0.5\n")

    for rank, ind in enumerate(hof):
        try:
            f, n_params = param_incl(
                [node.name for node in ind],
                unary_prim_funcs,
                bin_prim_funcs,
                arg_set,
                n_start=0
            )

            def make_model(f):
                def model(dummy_X, *thetas):
                    vals = f(data_arrays, thetas)
                    return vals if np.all(np.isfinite(vals)) else np.full(len(y_measured_global), 1e10)
                return model

            model = make_model(f)
            dummy_X = np.zeros(len(y_measured_global))
            p0 = [1.0] * n_params

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = curve_fit(model, dummy_X, y_measured_global, p0=p0, maxfev=5000)

            predicted = model(dummy_X, *popt)
            R2 = 1 - np.sum((y_measured_global - predicted)**2) / np.sum((y_measured_global - np.mean(y_measured_global))**2)
            params_str = ", ".join(f"θ[{i}]={v:.6f}" for i, v in enumerate(popt))

            print(f"{rank+1:2d}. {str(ind)}")
            print(f"    params : {params_str}")
            print(f"    R²     : {R2:.8f}  n_params={n_params}\n")

        except Exception as e:
            print(f"{rank+1:2d}. {str(ind)}")
            print(f"    FAILED : {e}\n")
    
    import matplotlib.pyplot as plt

    gens = range(len(stats["best"]))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(gens, stats["best"],  label="Best BIC",    color="green",  linewidth=2)
    ax1.plot(gens, stats["avg"],   label="Average BIC", color="blue",   linewidth=1.5, linestyle="--")
    ax1.plot(gens, stats["worst"], label="Worst BIC",   color="red",    linewidth=1,   linestyle=":")
    ax1.set_ylabel("BIC (lower = better)")
    ax1.set_title("GP Convergence — Feynman I.6.2")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(gens, stats["diversity"], color="purple", linewidth=2)
    ax2.set_ylabel("Diversity (unique exprs / pop size)")
    ax2.set_xlabel("Generation")
    ax2.set_ylim(0, 1)
    ax2.axhline(y=0.2, color="red", linestyle="--", alpha=0.5, label="Low diversity warning")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("gp_convergence.png", dpi=150)
    plt.show()
    print("Convergence plot saved to gp_convergence.png")