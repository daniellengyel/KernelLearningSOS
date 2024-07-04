import numpy as np
import random

import jax.numpy as jnp
import jax

from tqdm import tqdm

from Kernels import ext_kernel


def set_seed(seed):
    """
    Set the random seed for reproducibility in both standard Python and NumPy.

    Parameters:
    seed (int): The seed value to use for the random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)

def hausdorff_distance(x, y):
    HD = -float("inf")
    for i in range(len(x)):
        curr_min = jnp.min(jnp.linalg.norm(y - x[i], axis=1))
        HD = jnp.maximum(curr_min, HD)
    return HD

# def get_predict_func(k_func, X, Y):
#     K_XX = k_func(X[:, None], X) # + 1e-10 * np.eye(len(X)) # make_kernel_matrix(k_func, X, X)
#     K_XXinv_Y = jnp.linalg.solve(K_XX, Y)
    
#     def helper(x):
#         K_Xx = jnp.empty(len(X))
#         for i in range(len(X)):
#             K_Xx[i] = k_func(x, X[i])
#         return K_Xx.T @ K_XXinv_Y
        
#     return helper

def get_predict_func(k_func, theta, X, Y):
    # Calculate the kernel matrix and its inverse times Y
    K_XX = k_func(theta, X[:, None], X) # + 1e-10 * jnp.eye(len(X))
    K_XXinv_Y = jnp.linalg.solve(K_XX, Y)
    
    def helper(x):
        # Calculate the kernel values between the input x and all elements in X
        K_Xx = jax.vmap(lambda xi: k_func(theta, x, xi))(X)
        return K_Xx.T @ K_XXinv_Y
        
    return helper

def get_pred_funcs(thetas, k_func, X, Y):
    pred_funcs = []
    for i in range(len(thetas)):
        pred_funcs.append(get_predict_func(k_func, thetas[i], X, Y[:, i]))
    return pred_funcs
    
    

def gen_traj(curr_map, x0, N):

    x_curr = x0
    res = [x_curr]
    for _ in range(N):
        x_curr = curr_map(x_curr)
        res.append(x_curr)
    
    return res

def predict_series(pred_funcs, x0, N):
    d = len(x0)
    traj = jnp.zeros((N + 1, d))
    traj = traj.at[0, :].set(x0)
    for i in tqdm(range(1, N + 1)):
        for k in range(d):
            traj = traj.at[i, k].set(pred_funcs[k](traj[i - 1]))
    return traj
    
# ====== save and load ========
import pickle
import os

def save_res(dynamics_name, res):
    
    for k in res:
        with open(f"Results/{dynamics_name}/{k}.pkl", "wb") as f:
            pickle.dump(res[k], f)
        
        
def load_res(dynamics_name):
    res = {}
    for f_name in os.listdir(f"Results/{dynamics_name}"):
    
        with open(f"Results/{dynamics_name}/{f_name}", "rb") as f:
            res[f_name.split(".")[0]] = pickle.load(f)
        
    return res

        