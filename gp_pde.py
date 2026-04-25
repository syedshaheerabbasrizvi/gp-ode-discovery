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

# dN/dt = r*N*(1 - N/K)
# Ground truth structure: N - N^2 (with fitted r, r/K)
# ── SECTION 1: DATA GENERATOR (Allee Effect) ──
# ── SECTION 1: DATA GENERATOR (Quadratic Decay) ──
# ── SECTION 1: DATA GENERATOR (Pitchfork Bifurcation) ──
def data_gen_pitchfork(r=0.5, x0=0.1, noise=0.05, t=None):
    def ode(x, t):
        return r * x - x**3
    clean = odeint(ode, x0, t).flatten()
    
    scale = noise * np.mean(clean)
    noisy_x = clean + np.random.normal(scale=scale, size=clean.shape)
    return t, noisy_x

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
    # T maps string keys -> callable functions with signature f(vars, thetas)
    # Start by giving every leaf variable its own identity function
    T = {}
    for arg in arg_set:
        # Each leaf just looks up its value in the vars dict
        # default arg=arg captures current value immediately (avoids late binding)
        T[arg] = lambda vars, thetas, arg=arg: vars[arg]

    if len(exp_tree) > 1:
        k = 0
        n = n_start + 1  # theta[n_start] is the outer multiplier; inner thetas start at n_start+1

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
                    # + and - introduce a theta on the RIGHT operand only
                    Tn = lambda vars, thetas, L=L, R=R, op=op, n=n: op(L(vars, thetas), thetas[n] * R(vars, thetas))
                    n += 1
                else:
                    # * gets no theta
                    Tn = lambda vars, thetas, L=L, R=R, op=op: op(L(vars, thetas), R(vars, thetas))

                T[key] = Tn
                del exp_tree[k+2]
                del exp_tree[k+1]
                exp_tree[k] = key
                k = 0

            else:
                k += 1

    # Wrap the final collapsed term with the outer theta[n_start] multiplier
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
            # We universally expect 3 items: time, data, and the variable mapping
            for (t, measured_data, var_funcs) in experiments:
                
                # "state" is whatever the ODE is currently at (could be N, cA, x, etc.)
                def ode(state, t_val, var_funcs=var_funcs):
                    # Build the dictionary for the GP tree dynamically
                    vars_dict = {name: func(state) for name, func in var_funcs.items()}
                    val = f(vars_dict, thetas)
                    return np.clip(val, -1e5, 1e5)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ODEintWarning)
                    pred = odeint(ode, measured_data[0], t, rtol=1e-3, atol=1e-3, mxstep=100).flatten()
                    
                predictions.append(pred)
            return np.concatenate(predictions)

        t_all  = np.concatenate([exp[0] for exp in experiments])
        N_all = np.concatenate([exp[1] for exp in experiments])
        n_data = len(N_all)
        
        # CRITICAL FIX 2: Start with a tiny initial guess so the ODE doesn't explode
        p0 = [1e-4] * n_params  
        
        # CRITICAL FIX 3: Restrict parameters so curve_fit doesn't test insane values
        lower_bounds = [-100.0] * n_params
        upper_bounds = [100.0] * n_params

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            popt, _ = curve_fit(
                model, 
                t_all, 
                N_all, 
                p0=p0, 
                bounds=(lower_bounds, upper_bounds), 
                maxfev=1000 # Lower maxfev so it fails fast on bad equations
            )

        predicted = model(t_all, *popt)
        MSE = np.mean((predicted - N_all) ** 2)
        
        if not np.isfinite(MSE) or MSE <= 0:
            return (1e10,)
        
        tree_size = len(individual)
        BIC = n_data * np.log(MSE) + n_params * np.log(n_data)
        structural_weight = 3.0 
        structural_penalty = structural_weight * tree_size * np.log(n_data)
        modified_BIC = BIC + structural_penalty
        return (modified_BIC,) 

    except Exception as e:
        print(f"FAILED: {e} | expr: {individual[:3]}...") 
        return (1e10,)


# ─────────────────────────────────────────────
# SECTION 4: DEAP SETUP
# ─────────────────────────────────────────────

pset = gp.PrimitiveSet("dx_dt", arity=0)
pset.addPrimitive(operator.add, 2, name="+")
pset.addPrimitive(operator.sub, 2, name="-")
pset.addPrimitive(operator.mul, 2, name="*")
pset.addTerminal("x", name="x") # <--- Changed N to x

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

# ── SECTION 5: WIRING FITNESS INTO DEAP ──
bin_prim_funcs   = {"+": operator.add, "-": operator.sub, "*": operator.mul}
unary_prim_funcs = {}
arg_set = {"x"} # <--- Changed N to x
gp_config = {"unary_prim": unary_prim_funcs, "bin_prim": bin_prim_funcs, "arg_set": arg_set}

t = np.linspace(0, 10, 100)
t1, x1 = data_gen_pitchfork(r=0.5, x0=0.1, noise=0.05, t=t)
t2, x2 = data_gen_pitchfork(r=0.8, x0=0.2, noise=0.05, t=t)

experiments = [
    # Mapped the ODE state to the GP's new "x" variable!
    (t1, x1, {"x": lambda state: state}), 
    (t2, x2, {"x": lambda state: state}),
]

def evaluate_individual(ind):
    return fitness([node.name for node in ind], experiments, gp_config)
toolbox.register("evaluate", evaluate_individual)


# ─────────────────────────────────────────────
# SECTION 6: MAIN GP LOOP
# ─────────────────────────────────────────────

def run_gp(n_pop=100, n_gen=10, n_hof=10, cx_prob=0.6, mut_prob=0.2, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    pop = toolbox.population(n=n_pop)
    hof = tools.HallOfFame(n_hof)
    
    # --- NEW: STAT TRACKER ---
    stats_log = {"best": [], "avg": [], "worst": [], "diversity": []}

    with Pool() as pool:
        toolbox.register("map", pool.map)

        # Evaluate Gen 0
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        hof.update(pop)
        
        # Log Gen 0 Stats
        valid_fits = [ind.fitness.values[0] for ind in pop if ind.fitness.valid and ind.fitness.values[0] < 1e9]
        if valid_fits:
            stats_log["best"].append(np.min(valid_fits))
            stats_log["avg"].append(np.mean(valid_fits))
            stats_log["worst"].append(np.max(valid_fits))
        stats_log["diversity"].append(len(set(get_readable_expr(ind) for ind in pop)) / n_pop)
        
        logging.info("Gen   0 | initial population evaluated")

        for gen in tqdm(range(n_gen), desc="Evolving"):
            offspring = algorithms.varAnd(pop, toolbox, cxpb=cx_prob, mutpb=mut_prob)
            
            # Immigrants (Exploration Injection)
            n_immigrants = n_pop // 10
            immigrants = toolbox.population(n=n_immigrants)
            fitnesses = toolbox.map(toolbox.evaluate, immigrants)
            for ind, fit in zip(immigrants, fitnesses):
                ind.fitness.values = fit
            offspring = offspring + immigrants
            
            # Evaluate modified individuals
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid)
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

            pop = toolbox.select(offspring + pop, k=n_pop)
            hof.update(pop)

            # Log Generation Stats
            valid_fits = [ind.fitness.values[0] for ind in pop if ind.fitness.valid and ind.fitness.values[0] < 1e9]
            if valid_fits:
                best_val = np.min(valid_fits)
                stats_log["best"].append(best_val)
                stats_log["avg"].append(np.mean(valid_fits))
                stats_log["worst"].append(np.max(valid_fits))
                logging.info(f"Gen {gen+1:3d} | best score: {best_val:.3f}")
            else:
                stats_log["best"].append(0); stats_log["avg"].append(0); stats_log["worst"].append(0)
                logging.info(f"Gen {gen+1:3d} | best score: N/A")
                
            stats_log["diversity"].append(len(set(get_readable_expr(ind) for ind in pop)) / n_pop)

    return hof, stats_log

def get_readable_expr(individual):
    """Recursively converts a DEAP prefix tree into a human-readable infix math string."""
    nodes = list(individual) # Copy the list of nodes
    
    def build_string():
        node = nodes.pop(0)
        
        # If it's a leaf node (e.g., "N", "x", "c_A")
        if node.arity == 0:
            return node.name
            
        # If it's a unary operator (e.g., "sin", "cos", "exp")
        elif node.arity == 1:
            arg = build_string()
            return f"{node.name}({arg})"
            
        # If it's a binary operator (e.g., "+", "-", "*")
        elif node.arity == 2:
            left = build_string()
            right = build_string()
            return f"({left} {node.name} {right})"
            
    return build_string()

# ─────────────────────────────────────────────
# SECTION 7: RUN AND DISPLAY RESULTS
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.info("Running GP...")
    hof, stats_log = run_gp(n_pop=200, n_gen=5, n_hof=10)

    # Replaced print() with logging.info()
    logging.info("\n── Hall of Fame ──")
    for i, ind in enumerate(hof):
        logging.info(f"{i+1:2d}. BIC={ind.fitness.values[0]:.3f}  expr={get_readable_expr(ind)}")

    # ── Parameter fits ──
    logging.info("\n── Hall of Fame Parameter Fits ──")
    logging.info("Ground truth: dN/dt = r*N - (r/K)*N^2  (r=0.5, K=100 / r=0.3, K=150)\n")

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
                def model(t_concat, *thetas):
                    predictions = []
                    # Unpack the 3 items dynamically
                    for (t, measured_data, var_funcs) in experiments:
                        
                        def ode(state, t_val):
                            # Map dynamically instead of hardcoding "N"
                            vars_dict = {name: func(state) for name, func in var_funcs.items()}
                            val = f(vars_dict, thetas)
                            return np.clip(val, -1e5, 1e5)

                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", ODEintWarning)
                            pred = odeint(ode, measured_data[0], t, rtol=1e-3, atol=1e-3, mxstep=100).flatten()
                            
                        predictions.append(pred)
                    return np.concatenate(predictions)
                return model

            model = make_model(f)
            
            t_all  = np.concatenate([exp[0] for exp in experiments])
            N_all  = np.concatenate([exp[1] for exp in experiments])
            dummy_X = t_all
            p0 = [1e-4] * n_params
            lower_bounds = [-100.0] * n_params
            upper_bounds = [100.0] * n_params

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                # Curve fit optimization
                popt, _ = curve_fit(model, dummy_X, N_all, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=1000)

            predicted = model(dummy_X, *popt)
            MSE = np.mean((predicted - N_all)**2)
            R2  = 1 - np.sum((N_all - predicted)**2) / np.sum((N_all - np.mean(N_all))**2)
            params_str = ", ".join(f"θ[{i}]={v:.6f}" for i, v in enumerate(popt))

            # Log the final results to the file instead of the terminal
            logging.info(f"{rank+1:2d}. {get_readable_expr(ind)}")
            logging.info(f"    params : {params_str}")
            logging.info(f"    R²     : {R2:.6f}  MSE={MSE:.6f}  n_params={n_params}\n")
        except Exception as e:
            logging.info(f"{rank+1:2d}. {get_readable_expr(ind)}")
            logging.info(f"    FAILED : {e}\n")
        # ── PLOTTING STATS ──
    # ── PLOTTING STATS ──
    logging.info("Generating convergence and diversity plots...")
    
    gens = range(len(stats_log["best"]))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot 1: Convergence (Score tracking)
    ax1.plot(gens, stats_log["best"],  label="Best Score",    color="green",  linewidth=2)
    ax1.plot(gens, stats_log["avg"],   label="Average Score", color="blue",   linewidth=1.5, linestyle="--")
    
    # FIX 1: Removed the log scale so zeros/negatives don't break the graph
    ax1.set_ylabel("Score (Lower = Better)")
    ax1.set_title("GP Convergence (Exploitation vs. Exploration)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Diversity (Exploration tracking)
    ax2.plot(gens, stats_log["diversity"], label="Unique Equations Ratio", color="purple", linewidth=2)
    ax2.set_ylabel("Diversity Ratio")
    ax2.set_xlabel("Generation")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(y=0.15, color="red", linestyle=":", alpha=0.7, label="Stagnation Warning (15%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # FIX 2: Pop open the interactive window instead of saving to a file!
    logging.info("Opening interactive plot window...")
    plt.show()
        