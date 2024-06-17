import jax
import jax.numpy as jnp

from Kernels import make_kernels
from DataManager import get_curr_data
from Losses import rho_comp
from utils import set_seed

from tqdm import tqdm

# ======== Kernel Flow ========
def flow_rho_relative(theta_0, k_func, X_batch, y_batch, X_sub, y_sub, N=100, lr=0.001):
    theta = theta_0

    losses = []

    for _ in tqdm(range(N)):

        curr_loss = rho_comp(theta, k_func, X_batch, y_batch, X_sub, y_sub)
        curr_grad = jax.grad(rho_comp, argnums=0)(theta, k_func, X_batch, y_batch, X_sub, y_sub)

        losses.append(curr_loss)
        theta -= lr * curr_grad
    return theta, losses


def flow_rho_relative_SGD(theta_0, k_func, X_train, y_train, batch_size, sub_size, N=100, lr=0.001, do_subsample=False, seed=0):
    set_seed(seed)
    theta = theta_0

    losses = []

    X_batch, y_batch, X_sub, y_sub = get_curr_data(X_train, y_train, batch_size, sub_size)

    for _ in tqdm(range(N)):
        if do_subsample:
            X_batch, y_batch, X_sub, y_sub = get_curr_data(X_train, y_train, batch_size, sub_size)

        curr_loss = rho_comp(theta, k_func, X_batch, y_batch, X_sub, y_sub)
        curr_grad = jax.grad(rho_comp, argnums=0)(theta, k_func, X_batch, y_batch, X_sub, y_sub)

        losses.append(curr_loss)
        theta -= lr * curr_grad
    return theta, losses

    

# ========= IPM =========
# Hessian and Gradient for the Optimization Algorithm
def get_Hgrad(alpha, fs, Phi, lmbda, eps):
    n = len(fs)
    pre_inv_term = Phi @ jnp.diag(alpha) @ Phi.T + lmbda * jnp.eye(n)
    inv_pre_inv_term = jnp.linalg.inv(pre_inv_term)  # Precompute the inverse term for efficiency

    def compute_grad_element(i):
        return fs[i] - eps/n * (Phi[:, i].T @ inv_pre_inv_term @ Phi[:, i])

    grad = jax.vmap(compute_grad_element)(jnp.arange(n))

    return grad

def get_Hhess(alpha, fs, Phi, lmbda, eps):
    n = len(fs)
    pre_inv_term = Phi @ jnp.diag(alpha) @ Phi.T + lmbda * jnp.eye(n)
    inv_pre_inv_term = jnp.linalg.inv(pre_inv_term)  # Precompute the inverse term for efficiency

    def compute_hess_element(i, j):
        return eps / n * (Phi[:, i].T @ inv_pre_inv_term @ Phi[:, j]) ** 2

    hess = jax.vmap(
        lambda i: jax.vmap(lambda j: compute_hess_element(i, j))(jnp.arange(n))
    )(jnp.arange(n))

    return hess


def KernelSOS(alpha, thetas, fs, loss, N=50, lmbda=0.001, eps=1e-5, sig=0.1):
    res = []
    # Optimization Algorithm
    Phi, K = make_kernels(thetas, thetas, l=sig)
    for _ in tqdm(range(N)):

        n = len(alpha)

        hess = get_Hhess(alpha, fs, Phi, lmbda, eps)
        grad = get_Hgrad(alpha, fs, Phi, lmbda, eps)

        Hinv_grad = jnp.linalg.solve(hess, grad)
        Hinv_one = jnp.linalg.solve(hess, jnp.ones(n))
        delta = Hinv_grad - jnp.ones(n).T @ Hinv_grad / (jnp.ones(n).T @ Hinv_one) * Hinv_one
        newt_dec = delta.T @ hess @ delta
        alpha = alpha - 1/(1 + jnp.sqrt(n/ eps) * jnp.sqrt(newt_dec)) * delta
        
        theta_star = thetas.T @ alpha

        res.append([newt_dec, loss(theta_star)])
        
    theta_star = thetas.T @ alpha
    return theta_star, jnp.array(res), alpha