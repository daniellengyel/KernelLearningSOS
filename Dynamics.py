import numpy as np


# logistic map

def logistic_map(r):
    def helper(x):
        return r * x * (1 - x)
    return helper

def bernoulli_map():
    def helper(x):
        if x < 0.5:
            return 2 * x
        else:
            return 2 * x - 1

    return helper

def gen_traj(curr_map, x0, N):

    x_curr = x0
    res = []
    for _ in range(N):
        res.append(x_curr)
        x_curr = curr_map(x_curr)

    res.append(x_curr)
    
    return res

def lorentz_map(s=10, r=28, b=10/3.):
    
    def helper(x_vec):
        h = 0.01
        x, y, z = x_vec
        x1 = x + h * (s * (y - x))
        y1 = y + h * (r * x - y - x * z)
        z1 = z + h * (x * y - b * z)
        return [x1, y1, z1]
        
    return helper

def K_lorentz_one(theta):
    a0, a1, b, a2, s = theta
    def helper(x, y):
        xy_norm_squared = ((x - y)**2).sum(-1)
        return a0 + (a1 + np.sqrt(xy_norm_squared))**b + a2 * np.exp(-xy_norm_squared/s**2)
    return helper