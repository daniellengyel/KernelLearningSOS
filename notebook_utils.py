import numpy as np

import jax
import jax.numpy as jnp

from tqdm import tqdm

from Kernels import ext_kernel

from Optimizer import main_flow, main_sos
from DataManager import get_curr_data
from utils import predict_series, get_pred_funcs, set_seed, hausdorff_distance

jax.config.update("jax_enable_x64", True)


# ======== Getting the thetas =========
def get_thetas(X_train, Y_train, start_seed, flow_para, sos_para, num_theta=10):
    curr_seed = start_seed
    
    thetas_flow = []
    thetas_sos = []
    
    losses_flow = []
    losses_sos = []
    
    dim = X_train.shape[1]
    for i in range(num_theta):
        curr_thetas_flow = []
        curr_thetas_sos = []
        
        curr_losses_flow = []
        curr_losses_sos = []
        
        set_seed(curr_seed)
        X_batch, y_batch, X_sub, y_sub = get_curr_data(X_train, Y_train, len(X_train), len(X_train)//2)
        curr_seed += 1
        
        for y_idx in range(dim):
            curr_idx_losses_flow, curr_theta_star_flow = main_flow(y_idx, ext_kernel, X_batch, y_batch, X_sub, y_sub, curr_seed, flow_para['lr'], flow_para['num_steps'])
            curr_seed += 1

            curr_idx_losses_sos, curr_theta_star_sos = main_sos(y_idx, ext_kernel, X_batch, y_batch, X_sub, y_sub, curr_seed, sos_para["N_samples"], sos_para['N_steps'], sos_para['lmbda'], sos_para['eps'], sos_para['sig'])
            curr_seed += 1
            
            curr_thetas_flow.append(curr_theta_star_flow)
            curr_thetas_sos.append(curr_theta_star_sos)
            
            curr_losses_flow.append(curr_idx_losses_flow)
            curr_losses_sos.append(curr_idx_losses_sos)
            
            
            
        thetas_flow.append(curr_thetas_flow)
        thetas_sos.append(curr_thetas_sos)
        
        losses_flow.append(curr_losses_flow)
        losses_sos.append(curr_losses_sos)
        
        
        
    losses_flow = jnp.array(losses_flow)
    losses_sos = jnp.array(losses_sos)

            
    return thetas_flow, losses_flow, thetas_sos, losses_sos
        

# ======== Assesments ========
def get_NA_loss(pred_func, X_test, Y_test):
    res = []
    for x in tqdm(X_test):
        curr_res = []
        for k in range(len(pred_func)):
            curr_res.append(pred_func[k](x))
        res.append(curr_res)
        
    res = jnp.array(res)
    return jnp.mean(jnp.linalg.norm(res - Y_test, axis=1))

# Run this on train and test data
def do_test(thetas, x_vec_0_test, traj_test, X_train, Y_train, eps=1e-2):
    X_test = traj_test[:-1]
    Y_test = traj_test[1:]

    N_steps = len(X_test)
    
    HD_losses = []
    deviate_steps = []
    mse_losses = []
    
    for theta in thetas:
        pred_func = get_pred_funcs(theta, ext_kernel, X_train, Y_train)
        
        traj_pred = predict_series(pred_func, jnp.array(x_vec_0_test), N_steps)
        
        curr_HD = hausdorff_distance(traj_test, traj_pred)
        HD_losses.append(curr_HD)
        
        curr_deviate_step = np.argmin(np.linalg.norm(traj_test - traj_pred, axis=1) < eps)
        deviate_steps.append(curr_deviate_step)
        
        curr_mse_loss = get_NA_loss(pred_func, X_test, Y_test)
        mse_losses.append(curr_mse_loss)
    
    return HD_losses, deviate_steps, mse_losses
        
        
    