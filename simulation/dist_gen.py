import numpy as np

def rand_from_dist(f, x_min = 0, x_max = 1, n_x: int = 500, perturb: bool = True): # Generate a single random from distribution function
    # get starting params
    width = (x_max-x_min)
    dx = width / (n_x-1)
    # initialize linspaces
    x_arr = np.linspace(x_min, x_max, n_x)
    f_arr = np.array([f(x) for x in x_arr])
    if np.any(f_arr < 0):
        raise ValueError("Distribution function should not evaluate to a negative number.")
    # use numerical integration to determine the probability space
    sums = np.cumsum(f_arr*dx) # numerical integration basically
    # choose sample
    r = np.random.random()
    idx = np.searchsorted(sums, r*sums[-1])
    ret_val = x_arr[idx]
    if perturb:
        ret_val += np.random.uniform(-0.5*dx, 0.5*dx)
    return ret_val

def rfd_array(f, n: int = 1000, x_min = 0, x_max = 1, n_x: int = 200, perturb: bool = True): # Generate an array of n randoms from distribution function
    # get starting params
    width = (x_max-x_min)
    dx = width / (n_x-1)
    # initialize linspaces
    x_arr = np.linspace(x_min, x_max, n_x)
    f_arr = np.array([f(x) for x in x_arr])
    if np.any(f_arr < 0):
        raise ValueError("Distribution function should not evaluate to a negative number.")
    # use numerical integration to determine the probability space
    sums = np.cumsum(f_arr*dx)
    # create samples
    r_arr = np.random.random(n)
    # scale r from [0,1] to probability space [0, sums[-1]] and determine indices
    idx_arr = np.searchsorted(sums, r_arr * sums[-1])
    ret_arr = x_arr[idx_arr]
    # add perturbation if specified
    if perturb:
        ret_arr += np.random.uniform(-0.5*dx, 0.5*dx, n)
    return ret_arr

def evd_array(f, n: int = 1000, x_min = 0, x_max = 1, n_x: int = 200, perturb: bool = True): # Generate an array of n points spread evenly along the distribution
    # get starting params
    width = (x_max-x_min)
    dx = width / (n_x-1)
    # initialize linspaces
    x_arr = np.linspace(x_min, x_max, n_x)
    f_arr = np.array([f(x) for x in x_arr])
    if np.any(f_arr < 0):
        raise ValueError("Distribution function should not evaluate to a negative number.")
    # use numerical integration to determine the probability space
    sums = np.cumsum(f_arr*dx)
    # create samples
    r_arr = np.linspace(0, 1, n+2)[1:n+1]
    np.random.shuffle(r_arr)
    # scale r from [0,1] to probability space [0, sums[-1]] and determine indices
    idx_arr = np.searchsorted(sums, r_arr * sums[-1])
    ret_arr = x_arr[idx_arr]
    if perturb:
        ret_arr += np.random.uniform(-0.5*dx, 0.5*dx, n)
    return ret_arr
