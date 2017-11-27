def f(x, t, p):
    '''
    Objective value to decision x in scenario with reachability times t and detection probabilities p
    '''
    import numpy as np
    order = np.argsort(t)
    prob_fail = np.zeros((len(x)+1))
    prob_fail[0] = 1
    prob_fail[1:] = ((1 - p)**x)[order]
    cum_prob_fail = np.cumproduct(prob_fail)
    prob_succeed = (1 - prob_fail)[1:]
    return t.max() - (cum_prob_fail[:-1] * prob_succeed * t[order]).sum() - cum_prob_fail[-1]*t.max()

def gradient(x, t, p, log_p):
    '''
    Gradient of the objective with decision variables x and scenario t. p is the
    detection probabilities and log_p is log(1 - p) (elementwise). 
    
    Computations are performed via vectorized numpy operations for greater efficiency.
    '''
    import numpy as np
    order = np.argsort(t)
    x = x[order]
    p = p[order]
    log_p = log_p[order]
    t = t[order]
    p_fail = ((1 - p)**x)
    cum_prob_fail = np.cumproduct(p_fail)
    p_fail_offset = np.zeros((len(x)+1))
    p_fail_offset[0] = 1
    p_fail_offset[1:] = cum_prob_fail
    cum_sum = np.zeros((len(x)))
    cum_sum[-1] = 0
    intermediate = (p_fail_offset[:-1] * (1 - p_fail) * t)[1:]
    intermediate = intermediate[::-1]
    intermediate = np.cumsum(intermediate)
    intermediate = intermediate[::-1]
    cum_sum[:-1] = intermediate
    ordered_grad = t * log_p* cum_prob_fail - log_p * cum_sum - log_p*cum_prob_fail[-1]*t.max()
    ordered_grad[np.abs(ordered_grad) < 0.00001] = 0
    new_perm = np.zeros((len(order)), dtype=np.int)
    for i, val in enumerate(order):
        new_perm[val] = i
    return ordered_grad[new_perm]

def greedy_top_k(grad, elements, budget):
    '''
    Greedily select budget number of elements with highest weight according to
    grad
    '''
    import numpy as np
    combined = zip(elements, grad)
    combined.sort(key = lambda x: -x[1])
    indicator = np.zeros((len(elements)))
    for i in range(budget):
        indicator[elements.index(combined[i][0])] = 1
    return indicator
                
def cicm(g, mean_time, num_events):
    '''
    Generates scenarios from the continuous time independent cascade model.
    mean_time is the parameter of the exponential distributionn that edge times
    are drawn from. num_events is the number of scenarios to simulate. 
    
    Returns t: a num_events x (number of nodes) array where t[i,j] gives the time 
    contagion reaches node j in scenario i.
    '''
    import random
    import numpy as np
    import networkx as nx
    for u,v in g.edges():
        g[u][v]['weight'] = random.expovariate(mean_time)
    t = np.zeros((num_events, len(g)))
    for i in range(num_events):
        s = random.choice(g.nodes())
        dist = nx.shortest_path_length(g, source=s, weight = 'weight')
        for v in g.nodes():
            if v in dist:
                t[i, v] = dist[v]
            else:
                t[i, v] = np.inf
    t[t == np.inf] = t[t != np.inf].max()
    #remove scenarios where no sensor detects anything
    t = t[t.min(axis = 1) < t.max()]
    return t
     
def load_bwsn():
    '''
    Reads data output by the BWSN program. Outputs: a scenario array t, 
    as in cicm.
    '''
    import pandas as pd
    import numpy as np
    a = pd.read_csv('Network_1.inp.z1', index_col=False)
    num_nodes = len(a.columns[9:])
    num_samples = len(a)
    t = np.zeros((num_samples, num_nodes))
    for i, node in enumerate(a.columns[9:]):
        for j, entry in enumerate(a[node]):
            t[j, i] = entry
    t[t == t.max()] = t[t != t.max()].max()
    #remove scenarios where no sensor detects anything
    t = t[t.min(axis = 1) < t.max()]
    return t

def run_rascal(alpha, B, t, num_iter, pval):
    '''
    Runs RASCAL in the given scenario with CVaR parameter alpha, budget B,
    sampled scenarios (e.g. from cicm or load_bwsn) t, num_iter steps in the 
    optimization, and all nodes having equal detection probability pval.
    
    Returns: the final decision variables x and the final CVaR val.
    '''
    import numpy as np
    from functools import partial
    from rascal import rascal, cvar
    x = np.zeros((t.shape[1]))
    p = np.zeros((len(x)))
    p[:] = pval
    log_p = np.log(1 - p)
    objective = partial(f, p = p)
    FO = partial(gradient, p = p, log_p = log_p)
    U = np.zeros((len(x)))
    U[:] = np.inf
    L = np.zeros((len(x)))
    LO = partial(greedy, U = U, L = L, K = B)
    x, all_x = rascal(x, 0, num_iter, FO, LO, t, objective, t.max(), alpha, 0.1)
    val = cvar(x, objective, t, alpha)
    return x, val

def run_degree(g, t, B, alpha, pval):
    '''
    Selects the B nodes with highest degree and allocates each 1 unit of budget.
    '''
    from functools import partial
    import numpy as np
    from cvar_algo import cvar
    t = t[t.min(axis = 1) < t.max()]
    x = greedy_top_k([g.degree(v) for v in g.nodes()], range(len(g)), B)
    p = np.zeros((len(x)))
    p[:] = pval
    objective = partial(f, p = p)
    return x, cvar(x, objective, t, alpha)

def run_fw(alpha, B, t, pval):
    '''
    Runs the Frank-Wolfe algorithm to optimize the expected objective value.
    See run_rascal for parameter_values.
    '''
    import numpy as np
    from functools import partial
    from cvar_algo import sfw, cvar
    x = np.zeros((t.shape[1]))
    p = np.zeros((len(x)))
    p[:] = pval
    log_p = np.log(1 - p)
    FO = partial(gradient, p = p, log_p = log_p)
    U = np.zeros((len(x)))
    U[:] = np.inf
    L = np.zeros((len(x)))
    LO = partial(greedy, U = U, L = L, K = B)
    x = sfw(x, 50, FO, LO, t)
    objective = partial(f, p = p)
    val = cvar(x, objective, t, alpha)
    return x, val

def greedy(grad, U, L, K):
    '''
    Greedily find an allocation vector maximizing inner product with grad. Starts
    with L (a lower bound). U is an upper bound vector, and K a total budget 
    (constraint on how much L may be increased by). 
    '''
    import numpy as np
    elements = range(len(grad))
    combined = zip(elements, grad)
    combined.sort(key = lambda x: -x[1])
    sorted_groups = [x[0] for x in combined]
    nu = np.copy(L)
    curr = 0
    while (nu - L).sum() < K and curr < len(grad):
        amount_add = min([U[sorted_groups[curr]] - L[sorted_groups[curr]], K - (nu - L).sum()])
        nu[[sorted_groups[curr]]] += amount_add
        curr += 1
    return nu