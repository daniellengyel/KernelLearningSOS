import jax
import jax.numpy as jnp


# ====== Kernel utils ======
def make_kernels(X, Y, l=.1):
    
    n, dim = X.shape
    m = len(Y)
    K = gauss_kernel([1, l], X[:, None], Y)
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
# def gaussian_kernel(theta, x, y):
#     l = theta
#     return jnp.exp(-((x - y)**2).sum(-1) / l)

def poly_kernel(theta, x, y):
    c, d = theta
    xy_dot = (x * y).sum(-1)
    return (xy_dot + c)**d


# def logistic_kernel(theta, x, y):
#     a0, s1, s2, s3 = theta
#     xy_norm_squared = ((x - y)**2).sum(-1)
#     return a0 * jnp.exp(-s1 * jnp.sin(jnp.pi * s2 * jnp.sqrt(xy_norm_squared))**2) * jnp.exp(-xy_norm_squared/(s3**2)) 
        

def get_xy_norm_squared(x, y):
    return ((x - y)**2).sum(-1)

def get_xy_l1_norm(x, y):
    return (jnp.abs(x - y)).sum(-1)

def triangular_kernel(theta, x, y):
    gamma, sig = theta

    return gamma**2 * jnp.maximum(0, 1 - jnp.abs(x - y)/sig**2).sum(-1)

    # xy_norm_squared = get_xy_norm_squared(x, y) 
    # return gamma**2 * jnp.maximum(0, 1 - jnp.sqrt(xy_norm_squared)/sig**2)

# Got rid of typo where norm is squared. 

def gauss_kernel(theta, x, y):
    xy_norm_squared = get_xy_norm_squared(x, y) 
    gamma, sig = theta
    
    return gamma**2 * jnp.exp(-xy_norm_squared/sig**2)

def laplace_kernel(theta, x, y):
    xy_l1_norm = get_xy_l1_norm(x, y)
    gamma, sig = theta
    return gamma**2 * jnp.exp(-xy_l1_norm / sig**2)


# TODO: Have different parameters for each dimension. For now treat each the same. 
def locally_periodic_kernel(theta, x, y):
    gamma, sig0, sig1, sig2 = theta

    xy_norm_squared = get_xy_norm_squared(x, y)

    xy_sin_sum = (jnp.sin(jnp.pi * sig1**2 * jnp.abs(x - y))**2).sum(-1)

    return gamma**2 * jnp.exp(-sig0**2 * xy_sin_sum) * jnp.exp(-xy_norm_squared/sig2**2)

# DOESN'T WORK. WOULD WORK WITH INNER PRODUCTS (QUADRATIC KERNEL)
def squared_kernel(theta, x, y):
    gamma = theta
    xy_norm_squared = get_xy_norm_squared(x, y)
    return gamma * xy_norm_squared

def ext_kernel(theta, x, y):
    return triangular_kernel(theta[:2], x, y) + gauss_kernel(theta[2:4], x, y) + laplace_kernel(theta[4:6], x, y) + locally_periodic_kernel(theta[6:10], x, y)
