import jax
import jax.numpy as jnp


# ====== Kernel utils ======
def make_kernels(X, Y, l=.1):
    
    n, dim = X.shape
    m = len(Y)
    kernel_func = gaussian_kernel
    K = kernel_func(l, X[:, None], Y)
    K += 1e-14 * jnp.eye(len(K)) # TODO: REVIEW. just to be sure with Cholesky
    Phi = jnp.linalg.cholesky(K).T

    return Phi, K

# @partial(jax.jit, static_argnames=['k_func'])
def make_kernel_matrix(k_func, X, Y):
    K_XY = jnp.empty(shape=(len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            K_XY[i, j] = k_func(X[i], Y[j])
    return K_XY







# ======= Parameterized Kernel =======
# Kernels for Kernel SOS
def gaussian_kernel(theta, x, y):
    l = theta
    return jnp.exp(-((x - y)**2).sum(-1) / l)

def poly_kernel(theta, x, y):
    c, d = theta
    xy_dot = (x * y).sum(-1)
    return (xy_dot + c)**d


def logistic_kernel(theta, x, y):
    a0, s1, s2, s3 = theta
    xy_norm_squared = ((x - y)**2).sum(-1)
    return a0 * jnp.exp(-s1 * jnp.sin(jnp.pi * s2 * jnp.sqrt(xy_norm_squared))**2) * jnp.exp(-xy_norm_squared/(s3**2)) 
        