import jax.numpy as jnp
import numpy as np

def sample_selection(N, size):
    sample_indices = jnp.sort(np.random.choice(N, size, replace=False))
    return sample_indices

def batch_creation(N, batch_size, subsample_size):
    batch_indices = sample_selection(N, batch_size)
        
    # Sample from the mini-batch
    subsample_indices = sample_selection(batch_size, subsample_size)
    
    return batch_indices, subsample_indices

def get_curr_data(X, Y, batch_size, subsample_size):
    batch_indices, subsample_indices = batch_creation(len(X), batch_size, subsample_size)
    
    X_batch, y_batch = X[batch_indices], Y[batch_indices]
    X_sub, y_sub = X_batch[subsample_indices], y_batch[subsample_indices]
    
    return jnp.array(X_batch), jnp.array(y_batch), jnp.array(X_sub), jnp.array(y_sub)


# ======== Sampling ========
def sample_uniform(N, low_b, up_b, r):
    mid_val = (up_b + low_b)/2.
    length = (up_b - low_b) * r
    
    pts = (np.random.uniform(size=N) - 1/2.) * length + mid_val
    return pts

def sample_points(bounds, N, r=1):
    return np.array([sample_uniform(N, b[0], b[1], r) for b in bounds]).T #sample_uniform(N, -1, 1, r) # func_utils.get_points(np.zeros(self.d), N, fixed_bound=r*5)
