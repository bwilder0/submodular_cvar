def cvar(x, f, ys, alpha):
    '''
    Returns CVaR of a decision x. f is the objective function which is a parameter
    of both x and a random scenario y. ys is a list of scenarios. alpha is
    the parameter giving the degree of risk-aversion.
    '''
    import numpy as np
    fvals = [f(x, y) for y in ys]
    fvals.sort()
    return np.mean(fvals[:int(alpha*len(ys))])

def var(x, f, ys, alpha):
    '''
    Returns the value at risk of decision x.
    '''
    fvals = [f(x, y) for y in ys]
    fvals.sort()
    return fvals[int(alpha*len(ys))-1]

def rascal(x0, tau0, K, FO, LO, ys, f, M, alpha, u):
    '''
    Stochastic Frank-Wolfe algorithm.
    
    x0: initial point in R^n
    
    K: number of iterations
        
    FO: stochastic first-order oracle (returns unbiased estimate of gradient)
    
    LO: linear optimization oracle over the feasible set
    
    ys: list or array of scenarios, such that any entry can be passed to f
    
    f: objective function f(x, y)
    
    M: upper bound on the value of f
    
    alpha: CVaR parameter
    
    u: smoothing parameter
    '''
    import numpy as np
    x = np.array(x0)
    tau = tau0
    #tie breaker random noise
    r = np.random.random((len(ys))) * 0.00001
    all_x = []
    for k in range(K):
        print(k)
        all_x.append(x.copy())
        #compute gradient with respect to x
        grad = smooth_grad(x, tau, ys, f, FO, r, u, alpha)
        #update x
        v = LO(grad)
        x = x + (1./K)*v
        #update tau
        fvals = np.array([f(x,y) + r[j] for j,y in enumerate(ys)])
        tau = smooth_tau(fvals, alpha, u)    
        
    return x, all_x

def smooth_grad(x, tau, ys, f, FO, r, u, alpha):
    '''
    Computes smoothed estimate of gradient at x. See rascal for parameters.
    '''
    import numpy as np
    fvals = np.array([f(x, y) for y in ys])
    if u > 0:
        interval_length = tau + u - fvals
        interval_length = np.maximum(interval_length, 0)
        interval_length = np.minimum(interval_length, u)
        interval_length /= u
    else:
        interval_length = np.zeros((len(ys)))
        interval_length[fvals <= tau] = 1
    grad = np.zeros(len(x))
    for i,y in enumerate(ys):
        if fvals[i] <= tau + u:
            grad += interval_length[i] * FO(x, y)
    return 1./(u * alpha * len(ys)) * grad

def smooth_tau(fvals, alpha, u):
    '''
    Computes optimal value of tau over smoothed interval of size u via binary 
    search.
    '''
    upper = fvals.max()
    lower = fvals.min()
    tau = (upper - lower)/2
    while upper - lower > 0.001:
        if smooth_fraction_under(fvals, tau, u) > alpha:
            upper = tau
        else:
            lower = tau
        tau = (upper + lower)/2
    return tau

def smooth_fraction_under(fvals, tau, u):
    '''
    Helper function for smooth_tau
    '''
    import numpy as np
    if u > 0:
        interval_length = tau + u - fvals
        interval_length = np.maximum(interval_length, 0)
        interval_length = np.minimum(interval_length, u)
        interval_length /= u
    else:
        interval_length = np.zeros((len(fvals)))
        interval_length[fvals <= tau] = 1
    return interval_length.mean()


def sfw(x0, K, FO, LO, ys):
    '''
    Stochastic Frank-Wolfe algorithm for maximizing the expected value of f(x,y)
    
    x0: initial point in R^n
    
    K: number of iterations
        
    FO: stochastic first-order oracle (returns unbiased estimate of gradient)
    
    LO: linear optimization oracle over the feasible set
    
    ys: list or array of scenarios, such that any entry can be passed to f    
    '''
    import numpy as np
    x = np.array(x0)
    for i in range(K):
        grad = np.zeros((len(x0)))
        for j in range(len(ys)):
            grad += FO(x, ys[j])
        grad /= len(ys)
        v = LO(grad)
        x = x + (1./K)*v
    return x