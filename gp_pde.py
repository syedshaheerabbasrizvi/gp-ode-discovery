import numpy as np
import operator
import random
import warnings
import os, sys
import logging
from scipy.integrate import odeint
from scipy.optimize import curve_fit, OptimizeWarning
from deap import base, creator, tools, gp, algorithms
from tqdm import tqdm
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from scipy.integrate import odeint, ODEintWarning
import matplotlib.pyplot as plt

# ── Progress goes to gp_run.log, watch it with: tail -f gp_run.log ──
logging.basicConfig(
    filename="gp_run.log",
    filemode="w",
    format="%(message)s",
    level=logging.INFO
)


# ─────────────────────────────────────────────
# SECTION 1: DATA GENERATOR
# ─────────────────────────────────────────────

from python_benchmark import get_cstr_experiments

# ─────────────────────────────────────────────
# SECTION 2: PARAMETER INCLUSION (Algorithm 2)
# ─────────────────────────────────────────────

def param_incl(exp_tree, unary_prim, bin_prim, arg_set, n_start):
    """
    Takes an expression tree in prefix order and returns:
      - f: a callable f(vars, thetas) that evaluates dcA/dt
      - n_params: how many theta parameters were inserted

    vars   = dict of current variable values e.g. {"c_A": 0.3, "c_Af": 1.0, "c_A^2": 0.09}
    thetas = list of parameter values e.g. [0.5, 1.0, 2.0]
    """
    T = {}
    for arg in arg_set:
        T[arg] = lambda vars, thetas, arg=arg: vars[arg]

    if len(exp_tree) > 1:
        k = 0
        n = n_start + 1

        while len(exp_tree) > 1:

            # ── UNARY BRANCH ──
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

            # ── BINARY BRANCH ──
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
# SECTION 3: FITNESS EVALUATOR
# ─────────────────────────────────────────────

def fitness(individual, experiments, gp_config, n_start=0):
    try:
        f, n_params = param_incl(
            list(individual),
            gp_config["unary_prim"],
            gp_config["bin_prim"],
            gp_config["arg_set"],
            n_start
        )

        def model(t_concat, *thetas):
            predictions = []
            for (t, measured_data, var_funcs) in experiments:

                def ode(state, t_val, var_funcs=var_funcs):
                    vars_dict = {name: func(state) for name, func in var_funcs.items()}
                    val = f(vars_dict, thetas)
                    return np.clip(val, -1e5, 1e5)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ODEintWarning)
                    pred = odeint(ode, measured_data[0], t, rtol=1e-3, atol=1e-3, mxstep=100).flatten()

                predictions.append(pred)
            return np.concatenate(predictions)

        t_all  = np.concatenate([exp[0] for exp in experiments])
        N_all  = np.concatenate([exp[1] for exp in experiments])
        n_data = len(N_all)

        p0 = [1e-4] * n_params
        lower_bounds = [-100.0] * n_params
        upper_bounds = [100.0]  * n_params

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            popt, _ = curve_fit(
                model,
                t_all,
                N_all,
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
                maxfev=1000
            )

        predicted = model(t_all, *popt)
        MSE = np.mean((predicted - N_all) ** 2)

        if not np.isfinite(MSE) or MSE <= 0:
            return (1e10,)

        tree_size = len(individual)
        BIC = n_data * np.log(MSE) + n_params * np.log(n_data)
        structural_penalty = 3.0 * tree_size * np.log(n_data)
        return (BIC + structural_penalty,)

    except Exception as e:
        return (1e10,)


# ─────────────────────────────────────────────
# SECTION 4: DEAP SETUP
# ─────────────────────────────────────────────

pset = gp.PrimitiveSet("dcA_dt", arity=0)
pset.addPrimitive(operator.add, 2, name="+")
pset.addPrimitive(operator.sub, 2, name="-")
pset.addPrimitive(operator.mul, 2, name="*")
pset.addTerminal("c_A",   name="c_A")
pset.addTerminal("c_Af",  name="c_Af")
pset.addTerminal("c_A^2", name="c_A^2")

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr",       gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate",       gp.cxOnePoint)
toolbox.register("mutate",     gp.mutUniform, expr=toolbox.expr, pset=pset)
toolbox.register("select",     tools.selTournament, tournsize=3)
toolbox.decorate("mate",   gp.staticLimit(key=operator.attrgetter("height"), max_value=6))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6))


# ─────────────────────────────────────────────
# SECTION 5: WIRING FITNESS INTO DEAP
# ─────────────────────────────────────────────

bin_prim_funcs   = {"+": operator.add, "-": operator.sub, "*": operator.mul}
unary_prim_funcs = {}
arg_set          = {"c_A", "c_Af", "c_A^2"}
gp_config        = {"unary_prim": unary_prim_funcs, "bin_prim": bin_prim_funcs, "arg_set": arg_set}

experiments = get_cstr_experiments(noise=0.05, seed=0)

def evaluate_individual(ind):
    return fitness([node.name for node in ind], experiments, gp_config)

toolbox.register("evaluate", evaluate_individual)


# ─────────────────────────────────────────────
# SECTION 6: SEMANTIC SIMILARITY-BASED CROSSOVER (SSC)
#
# CSTR variable ranges: c_A in [0, 2], c_Af in [0.5, 2].
# ─────────────────────────────────────────────

SSC_ALPHA     = 1e-4   # lower SSD bound: subtrees must differ by at least this
SSC_BETA      = 0.4    # upper SSD bound: subtrees must not differ by more than this
SSC_MAX_TRIAL = 12     # attempts before falling back to standard crossover
SSC_N_POINTS  = 20     # random sample points for semantic evaluation (i am using all the values as of paper)

_SSC_CA_RANGE  = (0.0,  2.0) #these are values for testing the similarity so just putting the constraints on dataa poijts.
_SSC_CAF_RANGE = (0.5,  2.0)


def _sample_subtree_semantics_cstr(node_names, n_points=SSC_N_POINTS):
    rng = np.random.default_rng()
    ca_s  = rng.uniform(*_SSC_CA_RANGE,  size=n_points)
    caf_s = rng.uniform(*_SSC_CAF_RANGE, size=n_points)

    sample_arrays = {
        "c_A":   ca_s,
        "c_Af":  caf_s,
        "c_A^2": ca_s ** 2,
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
    """Mean absolute difference (Section 3.1 of the SSC paper)."""
    return float(np.mean(np.abs(sem1 - sem2)))


def semantic_similarity_crossover(ind1, ind2,
                                   alpha=SSC_ALPHA,
                                   beta=SSC_BETA,
                                   max_trials=SSC_MAX_TRIAL,
                                   n_points=SSC_N_POINTS):

    if len(ind1) < 2 or len(ind2) < 2:
        return gp.cxOnePoint(ind1, ind2)

    for _ in range(max_trials):
        # Random crossover points (skip root at index 0)
        idx1 = random.randint(1, len(ind1) - 1)
        idx2 = random.randint(1, len(ind2) - 1)

        # Get subtree boundaries as plain integers
        sl1 = ind1.searchSubtree(idx1)
        sl2 = ind2.searchSubtree(idx2)
        s1, e1 = sl1.start, sl1.stop
        s2, e2 = sl2.start, sl2.stop

        names1 = [node.name for node in list(ind1)[s1:e1]]
        names2 = [node.name for node in list(ind2)[s2:e2]]

        sem1 = _sample_subtree_semantics_cstr(names1, n_points)
        sem2 = _sample_subtree_semantics_cstr(names2, n_points)

        if sem1 is None or sem2 is None:
            continue

        dist = _ssd(sem1, sem2)

        if alpha < dist < beta:
            # Semantically similar enough — perform the swap
            nodes1 = list(ind1)
            nodes2 = list(ind2)

            sub1 = nodes1[s1:e1]
            sub2 = nodes2[s2:e2]

            new_nodes1 = nodes1[:s1] + sub2 + nodes1[e1:]
            new_nodes2 = nodes2[:s2] + sub1 + nodes2[e2:]

            if not new_nodes1 or not new_nodes2:
                continue

            ind1[0:len(ind1)] = new_nodes1
            ind2[0:len(ind2)] = new_nodes2

            del ind1.fitness.values
            del ind2.fitness.values
            return ind1, ind2

    # Fallback after max_trials failures
    return gp.cxOnePoint(ind1, ind2)


# ─────────────────────────────────────────────
# SECTION 7: MAIN GP LOOP
# ─────────────────────────────────────────────

def get_readable_expr(individual):
    """Recursively converts a DEAP prefix tree into a human-readable infix math string."""
    nodes = list(individual)

    def build_string():
        node = nodes.pop(0)
        if node.arity == 0:
            return node.name
        elif node.arity == 1:
            arg = build_string()
            return f"{node.name}({arg})"
        elif node.arity == 2:
            left  = build_string()
            right = build_string()
            return f"({left} {node.name} {right})"

    return build_string()


def run_gp(n_pop=100, n_gen=10, n_hof=10, cx_prob=0.6, mut_prob=0.2,
           seed=42, use_ssc=False):
    random.seed(seed)
    np.random.seed(seed)

    mode_label = "SSC" if use_ssc else "Standard"

    # Re-register mate operator and reapply decorators
    if use_ssc:
        toolbox.register("mate", semantic_similarity_crossover)
    else:
        toolbox.register("mate", gp.cxOnePoint)
    toolbox.decorate("mate",   gp.staticLimit(key=operator.attrgetter("height"), max_value=6))

    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6))

    pop = toolbox.population(n=n_pop)
    hof = tools.HallOfFame(n_hof)
    stats_log = {"best": [], "avg": [], "worst": [], "diversity": []}

    with Pool() as pool:
        toolbox.register("map", pool.map)

        # Evaluate Gen 0
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        hof.update(pop)

        valid_fits = [ind.fitness.values[0] for ind in pop
                      if ind.fitness.valid and ind.fitness.values[0] < 1e9]
        if valid_fits:
            stats_log["best"].append(np.min(valid_fits))
            stats_log["avg"].append(np.mean(valid_fits))
            stats_log["worst"].append(np.max(valid_fits))
        stats_log["diversity"].append(
            len(set(get_readable_expr(ind) for ind in pop)) / n_pop
        )
        logging.info(f"[{mode_label}] Gen   0 | initial population evaluated")

        for gen in tqdm(range(n_gen), desc=f"Evolving ({mode_label})"):
            offspring = algorithms.varAnd(pop, toolbox, cxpb=cx_prob, mutpb=mut_prob)

            # Immigrants (exploration injection)
            n_immigrants = n_pop // 10
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

            valid_fits = [ind.fitness.values[0] for ind in pop
                          if ind.fitness.valid and ind.fitness.values[0] < 1e9]
            if valid_fits:
                best_val = np.min(valid_fits)
                stats_log["best"].append(best_val)
                stats_log["avg"].append(np.mean(valid_fits))
                stats_log["worst"].append(np.max(valid_fits))
                logging.info(f"[{mode_label}] Gen {gen+1:3d} | best score: {best_val:.3f}")
            else:
                stats_log["best"].append(0)
                stats_log["avg"].append(0)
                stats_log["worst"].append(0)
                logging.info(f"[{mode_label}] Gen {gen+1:3d} | best score: N/A")

            stats_log["diversity"].append(
                len(set(get_readable_expr(ind) for ind in pop)) / n_pop
            )

    return hof, stats_log


# ─────────────────────────────────────────────
# SECTION 8: PRINT HOF WITH PARAMETER FITS
# ─────────────────────────────────────────────

def print_hof(hof, label):
    logging.info(f"\n── Hall of Fame [{label}] ──")
    for i, ind in enumerate(hof):
        logging.info(f"{i+1:2d}. BIC={ind.fitness.values[0]:.3f}  expr={get_readable_expr(ind)}")

    logging.info(f"\n── Hall of Fame Parameter Fits [{label}] ──")
    logging.info("Ground truth: dcA/dt = 0.5*(cAf - cA) - cA^2")

    for rank, ind in enumerate(hof):
        try:
            f, n_params = param_incl(
                [node.name for node in ind],
                unary_prim_funcs, bin_prim_funcs, arg_set, n_start=0
            )

            def make_model(f):
                def model(t_concat, *thetas):
                    predictions = []
                    for (t, measured_data, var_funcs) in experiments:
                        def ode(state, t_val):
                            vars_dict = {name: func(state) for name, func in var_funcs.items()}
                            val = f(vars_dict, thetas)
                            return np.clip(val, -1e5, 1e5)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", ODEintWarning)
                            pred = odeint(ode, measured_data[0], t,
                                          rtol=1e-3, atol=1e-3, mxstep=100).flatten()
                        predictions.append(pred)
                    return np.concatenate(predictions)
                return model

            model   = make_model(f)
            t_all   = np.concatenate([exp[0] for exp in experiments])
            N_all   = np.concatenate([exp[1] for exp in experiments])
            p0      = [0.5] * n_params

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                popt, _ = curve_fit(model, t_all, N_all,
                                    p0=p0,
                                    bounds=([-100.0]*n_params, [100.0]*n_params),
                                    maxfev=1000)

            predicted  = model(t_all, *popt)
            MSE        = np.mean((predicted - N_all) ** 2)
            R2         = 1 - np.sum((N_all - predicted)**2) / \
                             np.sum((N_all - np.mean(N_all))**2)
            params_str = ", ".join(f"θ[{i}]={v:.6f}" for i, v in enumerate(popt))

            logging.info(f"{rank+1:2d}. {get_readable_expr(ind)}")
            logging.info(f"    params : {params_str}")
            logging.info(f"    R*R     : {R2:.6f}  MSE={MSE:.6f}  n_params={n_params}\n")
        except Exception as e:
            logging.info(f"{rank+1:2d}. {get_readable_expr(ind)}")
            logging.info(f"    FAILED : {e}\n")


# ─────────────────────────────────────────────
# SECTION 9: RUN AND DISPLAY RESULTS
# ─────────────────────────────────────────────

if __name__ == "__main__":
    GP_PARAMS = dict(n_pop=200, n_gen=5, n_hof=10, cx_prob=0.6, mut_prob=0.2)

    logging.info("=" * 60)
    logging.info("Running GP with SEMANTIC SIMILARITY-BASED CROSSOVER (SSC) ...")
    logging.info("=" * 60)
    hof_ssc, stats_ssc = run_gp(**GP_PARAMS, seed=42, use_ssc=True)

    logging.info("=" * 60)
    logging.info("Running GP with STANDARD CROSSOVER ...")
    logging.info("=" * 60)
    hof_sc, stats_sc = run_gp(**GP_PARAMS, seed=42, use_ssc=False)

    print_hof(hof_sc,  "Standard Crossover")
    print_hof(hof_ssc, "SSC")

    # ── Comparison summary ──
    best_sc  = hof_sc[0].fitness.values[0]
    best_ssc = hof_ssc[0].fitness.values[0]
    logging.info("\n" + "=" * 60)
    logging.info("COMPARISON SUMMARY")
    logging.info("=" * 60)
    logging.info(f"  Standard — Best BIC : {best_sc:.4f}")
    logging.info(f"  SSC      — Best BIC : {best_ssc:.4f}")
    logging.info(f"  Winner (lower is better) : {'SSC' if best_ssc < best_sc else 'Standard'}")
    logging.info(f"  Final diversity-SC : {stats_sc['diversity'][-1]:.3f}")
    logging.info(f"  Final diversity-SSC: {stats_ssc['diversity'][-1]:.3f}")

    # ── PLOTTING ──
    import matplotlib.gridspec as gridspec

    gens = range(len(stats_sc["best"]))
    fig  = plt.figure(figsize=(14, 10))
    gs   = gridspec.GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(gens, stats_sc["best"],  label="SC  — Best Score",  color="blue",   linewidth=2)
    ax1.plot(gens, stats_ssc["best"], label="SSC — Best Score",  color="orange", linewidth=2, linestyle="--")
    ax1.set_ylabel("Best Score (lower = better)")
    ax1.set_title("Best Score: Standard vs SSC")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(gens, stats_sc["avg"],  label="SC  — Avg Score",  color="blue",   linewidth=2)
    ax2.plot(gens, stats_ssc["avg"], label="SSC — Avg Score",  color="orange", linewidth=2, linestyle="--")
    ax2.set_ylabel("Average Score")
    ax2.set_title("Average Score: Standard vs SSC")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(gens, stats_sc["diversity"],  label="SC  — Diversity", color="blue",   linewidth=2)
    ax3.plot(gens, stats_ssc["diversity"], label="SSC — Diversity", color="orange", linewidth=2, linestyle="--")
    ax3.axhline(y=0.15, color="red", linestyle=":", alpha=0.7, label="Stagnation warning (15%)")
    ax3.set_ylabel("Diversity Ratio (unique exprs / pop)")
    ax3.set_xlabel("Generation")
    ax3.set_ylim(0, 1.05)
    ax3.set_title("Population Diversity: Standard vs SSC")
    ax3.legend(); ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(gens, stats_ssc["best"],  label="SSC Best",  color="green",  linewidth=2)
    ax4.plot(gens, stats_ssc["avg"],   label="SSC Avg",   color="blue",   linewidth=1.5, linestyle="--")
    ax4.plot(gens, stats_ssc["worst"], label="SSC Worst", color="red",    linewidth=1,   linestyle=":")
    ax4.set_ylabel("Score")
    ax4.set_xlabel("Generation")
    ax4.set_title("SSC Convergence Detail")
    ax4.legend(); ax4.grid(True, alpha=0.3)

    plt.suptitle(
        "GP Symbolic Regression — CSTR (dcA/dt)\n"
        "Standard Crossover vs Semantic Similarity-based Crossover (SSC)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig("gp_comparison_sc_vs_ssc_cstr.png", dpi=150)
    logging.info("\nPlot saved to gp_comparison_sc_vs_ssc_cstr.png")
    plt.show()