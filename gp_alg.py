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
import copy
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

def get_logger(name, filepath):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath, mode="w")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    return logger

logger_sc  = get_logger("sc",  "gp_standard_crossover.log")
logger_ssc = get_logger("ssc", "gp_semantic_crossover.log")


# ─────────────────────────────────────────────
# SECTION 1: PARAM INCLUSION (unchanged)
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
# SECTION 2: FITNESS (unchanged)
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
                vals = f(data_arrays, thetas)
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
        BIC = n_data * np.log(MSE + 1e-30) + n_params * np.log(n_data) * 3
        return (BIC,)

    except Exception:
        return (1e10,)


# ─────────────────────────────────────────────
# SECTION 3: DEAP SETUP (unchanged)
# ─────────────────────────────────────────────

pset = gp.PrimitiveSet("f", arity=0)
pset.addPrimitive(operator.add,  2, name="+")
pset.addPrimitive(operator.sub,  2, name="-")
pset.addPrimitive(operator.mul,  2, name="*")
pset.addPrimitive(safe_div,      2, name="/")
pset.addPrimitive(safe_exp,      1, name="exp")
pset.addPrimitive(safe_sq,       1, name="sq")

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
# SECTION 4: DATA AND FITNESS (unchanged)
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
    raw = np.loadtxt(filepath)
    if n_samples is not None and n_samples < len(raw):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(raw), size=n_samples, replace=False)
        raw = raw[idx]
    X = raw[:, :-1]
    y = raw[:, -1]
    if noise_frac > 0.0:
        rng = np.random.default_rng(seed)
        y = y + rng.normal(scale=noise_frac * np.std(y), size=len(y))
    return X, y

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
# SECTION 5: SEMANTIC SIMILARITY-BASED CROSSOVER (SSC)

SSC_ALPHA     = 1e-4   # lower SSD bound: subtrees must differ by at least this
SSC_BETA      = 0.4    # upper SSD bound: subtrees must not differ by more than this
SSC_MAX_TRIAL = 12     # attempts before falling back to standard crossover
SSC_N_POINTS  = 20     # random sample points for semantic evaluation

_SSC_THETA_RANGE = (-3.0, 3.0)
_SSC_SIGMA_RANGE = (0.5,  3.0)


def _sample_subtree_semantics(node_names, n_points=SSC_N_POINTS):
    rng = np.random.default_rng()
    theta_s = rng.uniform(*_SSC_THETA_RANGE, size=n_points)
    sigma_s = rng.uniform(*_SSC_SIGMA_RANGE, size=n_points)

    sample_arrays = {
        "theta":   theta_s,
        "sigma":   sigma_s,
        "theta^2": theta_s**2,
        "sigma^2": sigma_s**2,
    }
    try:
        f, n_params = param_incl(
            list(node_names),       # param_incl mutates its input — must pass a copy
            unary_prim_funcs,
            bin_prim_funcs,
            arg_set,
            n_start=0
        )
        thetas = np.ones(n_params + 1)
        vals = f(sample_arrays, thetas)
        if not np.all(np.isfinite(vals)):
            return None
        return vals
    except Exception:
        return None


def _ssd(sem1, sem2):
    """Mean absolute difference"""
    return float(np.mean(np.abs(sem1 - sem2)))


def semantic_similarity_crossover(ind1, ind2,
                                   alpha=SSC_ALPHA,
                                   beta=SSC_BETA,
                                   max_trials=SSC_MAX_TRIAL,
                                   n_points=SSC_N_POINTS):
    if len(ind1) < 2 or len(ind2) < 2:
        return gp.cxOnePoint(ind1, ind2)

    for _ in range(max_trials):
        # Random crossover points
        idx1 = random.randint(1, len(ind1) - 1)
        idx2 = random.randint(1, len(ind2) - 1)

        # Get subtree boundaries as plain integers
        sl1 = ind1.searchSubtree(idx1)
        sl2 = ind2.searchSubtree(idx2)
        s1, e1 = sl1.start, sl1.stop 
        s2, e2 = sl2.start, sl2.stop 

        # Evaluate semantics of each subtree
        names1 = [node.name for node in list(ind1)[s1:e1]]
        names2 = [node.name for node in list(ind2)[s2:e2]]

        sem1 = _sample_subtree_semantics(names1, n_points)
        sem2 = _sample_subtree_semantics(names2, n_points)

        if sem1 is None or sem2 is None:
            continue

        dist = _ssd(sem1, sem2)

        if alpha < dist < beta:
            nodes1 = list(ind1)
            nodes2 = list(ind2)

            sub1 = nodes1[s1:e1] 
            sub2 = nodes2[s2:e2]   
            
            new_nodes1 = nodes1[:s1] + sub2 + nodes1[e1:]
            new_nodes2 = nodes2[:s2] + sub1 + nodes2[e2:]

            if not new_nodes1 or not new_nodes2:
                continue

            # Assign back — safe because we pass a complete plain list
            ind1[0:len(ind1)] = new_nodes1
            ind2[0:len(ind2)] = new_nodes2

            del ind1.fitness.values
            del ind2.fitness.values
            return ind1, ind2

    # Fallback after max_trials failures
    return gp.cxOnePoint(ind1, ind2)


# ─────────────────────────────────────────────
# SECTION 6: GP LOOP
# ─────────────────────────────────────────────

def run_gp(n_pop=200, n_gen=30, n_hof=10, cx_prob=0.6, mut_prob=0.2,
           seed=42, use_ssc=False):
    random.seed(seed)
    np.random.seed(seed)

    logger     = logger_ssc if use_ssc else logger_sc
    mode_label = "SSC" if use_ssc else "Standard"

    # Re-register mate and reapply decorators
    if use_ssc:
        toolbox.register("mate", semantic_similarity_crossover)
    else:
        toolbox.register("mate", gp.cxOnePoint)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6))
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=20))

    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=20))

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
        logger.info(f"[{mode_label}] Gen 0 | initial population evaluated")

        for gen in tqdm(range(n_gen), desc=f"Evolving ({mode_label})"):
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
            logger.info(f"[{mode_label}] Gen {gen+1:3d} | best BIC: {best:.3f}")

    return hof, {
        "best":      history_best,
        "avg":       history_avg,
        "worst":     history_worst,
        "diversity": history_diversity,
    }


# ─────────────────────────────────────────────
# SECTION 7: RESULTS AND COMPARISON PLOT
# ─────────────────────────────────────────────

def print_hof(hof, label):
    print(f"\n── Hall of Fame [{label}] ──")
    for i, ind in enumerate(hof):
        print(f"{i+1:2d}. BIC={ind.fitness.values[0]:.3f}  expr={str(ind)}")

    print(f"\n── Parameter Fits [{label}] ──")
    print(f"Ground truth: 1/sqrt(2π) ≈ {1/np.sqrt(2*np.pi):.6f}\n")

    for rank, ind in enumerate(hof):
        try:
            f, n_params = param_incl(
                [node.name for node in ind],
                unary_prim_funcs, bin_prim_funcs, arg_set, n_start=0
            )
            def make_model(f):
                def model(dummy_X, *thetas):
                    vals = f(data_arrays, thetas)
                    return vals if np.all(np.isfinite(vals)) else np.full(len(y_measured_global), 1e10)
                return model
            model   = make_model(f)
            dummy_X = np.zeros(len(y_measured_global))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(model, dummy_X, y_measured_global,
                                    p0=[1.0]*n_params, maxfev=5000)
            predicted = model(dummy_X, *popt)
            R2 = 1 - np.sum((y_measured_global - predicted)**2) / \
                     np.sum((y_measured_global - np.mean(y_measured_global))**2)
            params_str = ", ".join(f"θ[{i}]={v:.6f}" for i, v in enumerate(popt))
            print(f"{rank+1:2d}. {str(ind)}")
            print(f"    params : {params_str}")
            print(f"    R²     : {R2:.8f}  n_params={n_params}\n")
        except Exception as e:
            print(f"{rank+1:2d}. {str(ind)}")
            print(f"    FAILED : {e}\n")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    GP_PARAMS = dict(n_pop=500, n_gen=50, n_hof=10, cx_prob=0.5, mut_prob=0.4)
    print("\n" + "=" * 60)
    print("Running GP with SEMANTIC SIMILARITY-BASED CROSSOVER (SSC) ...")
    print("=" * 60)
    hof_ssc, stats_ssc = run_gp(**GP_PARAMS, seed=42, use_ssc=True)

    print("=" * 60)
    print("Running GP with STANDARD CROSSOVER ...")
    print("=" * 60)
    hof_sc, stats_sc = run_gp(**GP_PARAMS, seed=42, use_ssc=False)


    print_hof(hof_sc,  "Standard Crossover")
    print_hof(hof_ssc, "SSC")

    best_sc  = hof_sc[0].fitness.values[0]
    best_ssc = hof_ssc[0].fitness.values[0]
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  Standard — Best BIC : {best_sc:.4f}")
    print(f"  SSC      — Best BIC : {best_ssc:.4f}")
    print(f"  Winner (lower is better) : {'SSC' if best_ssc < best_sc else 'Standard'}")
    print(f"  Final diversity — SC : {stats_sc['diversity'][-1]:.3f}")
    print(f"  Final diversity — SSC: {stats_ssc['diversity'][-1]:.3f}")

    gens = range(len(stats_sc["best"]))
    fig  = plt.figure(figsize=(14, 10))
    gs   = gridspec.GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(gens, stats_sc["best"],  label="SC  — Best BIC",  color="blue",   linewidth=2)
    ax1.plot(gens, stats_ssc["best"], label="SSC — Best BIC",  color="orange", linewidth=2, linestyle="--")
    ax1.set_ylabel("Best BIC (lower = better)")
    ax1.set_title("Best BIC: Standard vs SSC")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(gens, stats_sc["avg"],  label="SC  — Avg BIC",  color="blue",   linewidth=2)
    ax2.plot(gens, stats_ssc["avg"], label="SSC — Avg BIC",  color="orange", linewidth=2, linestyle="--")
    ax2.set_ylabel("Average BIC")
    ax2.set_title("Average BIC: Standard vs SSC")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(gens, stats_sc["diversity"],  label="SC  — Diversity", color="blue",   linewidth=2)
    ax3.plot(gens, stats_ssc["diversity"], label="SSC — Diversity", color="orange", linewidth=2, linestyle="--")
    ax3.axhline(y=0.2, color="red", linestyle=":", alpha=0.5, label="Low diversity threshold")
    ax3.set_ylabel("Diversity (unique exprs / pop)")
    ax3.set_xlabel("Generation")
    ax3.set_ylim(0, 1)
    ax3.set_title("Population Diversity: Standard vs SSC")
    ax3.legend(); ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(gens, stats_ssc["best"],  label="SSC Best",  color="green",  linewidth=2)
    ax4.plot(gens, stats_ssc["avg"],   label="SSC Avg",   color="blue",   linewidth=1.5, linestyle="--")
    ax4.plot(gens, stats_ssc["worst"], label="SSC Worst", color="red",    linewidth=1,   linestyle=":")
    ax4.set_ylabel("BIC")
    ax4.set_xlabel("Generation")
    ax4.set_title("SSC Convergence Detail")
    ax4.legend(); ax4.grid(True, alpha=0.3)

    plt.suptitle(
        "GP Symbolic Regression — Feynman I.6.2\n"
        "Standard Crossover vs Semantic Similarity-based Crossover (SSC)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig("gp_comparison_sc_vs_ssc.png", dpi=150)
    plt.show()
    print("\nPlot saved to gp_comparison_sc_vs_ssc.png")