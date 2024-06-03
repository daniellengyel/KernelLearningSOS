import jax
import jax.numpy as jnp

# ======== Kernel Flow ========

def flow_rho(X, Y, N_b, N_c, delta):

    pass
    


# ========= IPM =========
# Hessian and Gradient for the Optimization Algorithm
def get_Hgrad(alpha, fs, Phi, lmbda, eps):
    n = len(fs)
    grad = jnp.empty(n)
    pre_inv_term = Phi @ jnp.diag(alpha) @ Phi.T + lmbda * jnp.eye(n)
    for i in range(n):
        
        grad[i] = fs[i] - eps/n * Phi[:, i].T @ jnp.linalg.solve(pre_inv_term, Phi[:, i])
    return grad

def get_Hhess(alpha, fs, Phi, lmbda, eps):
    n = len(fs)
    hess = jnp.empty(shape=(n, n))
    pre_inv_term = Phi @ jnp.diag(alpha) @ Phi.T + lmbda * jnp.eye(n)
    for i in range(n):
        for j in range(n):
            hess[i, j] = eps/n * (Phi[:, i].T @ jnp.linalg.solve(pre_inv_term, Phi[:, j]))**2
    return hess


def IPM()
    # Optimization Algorithm
    Phi, K = make_kernels(thetas, thetas)
    for _ in range(20):

        lmbda = 0.0001
        eps = 0.1
        n = len(alpha)

        hess = get_Hhess(alpha, fs, Phi, lmbda, eps)
        grad = get_Hgrad(alpha, fs, Phi, lmbda, eps)
        

        Hinv_grad = jnp.linalg.solve(hess, grad)
        Hinv_one = jnp.linalg.solve(hess, jnp.ones(n))
        delta = Hinv_grad - jnp.ones(n).T @ Hinv_grad / (jnp.ones(n).T @ Hinv_one) * Hinv_one
        newt_dec = delta.T @ hess @ delta
        print(newt_dec)
        alpha = alpha - 1/(1 + jnp.sqrt(n/ eps) * jnp.sqrt(newt_dec)) * delta
    theta_star = thetas.T @ alpha

    return theta_star