import jax
import jax.numpy as jnp

def rho_comp(theta, k_func, X_full, y_full, X_sub, y_sub):
    K_full = k_func(theta, X_full[:, None], X_full) # make_kernel_matrix(k_func, Xb, Xb)
    # Kb += 1e-12 * jnp.eye(len(Kb)) # TODO: REVIEW. just to be sure with Cholesky

    K_sub = k_func(theta, X_sub[:, None], X_sub) # make_kernel_matrix(k_func, Xc, Xc)
    # Kc += 1e-12 * jnp.eye(len(Kc)) # TODO: REVIEW. just to be sure with Cholesky

    return 1 - (y_sub @ jnp.linalg.solve(K_sub, y_sub))/(y_full @ jnp.linalg.solve(K_full, y_full))


def MMD_comp(theta, k_func, X1, X2):
    res = 0
    m = len(X1)
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            res += k_func(theta, X1[i], X1[j])
            res += k_func(theta, X2[i], X2[j])
            res -= k_func(theta, X1[i], X2[j])
            res -= k_func(theta, X1[j], X2[i])
    return res / (m * (m - 1))