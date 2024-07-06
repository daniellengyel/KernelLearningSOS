import numpy as np

import jax
import jax.numpy as jnp

from tqdm import tqdm

from Kernels import ext_kernel

from Optimizer import main_flow, main_sos
from DataManager import get_curr_data
from utils import predict_series, get_pred_funcs, set_seed, hausdorff_distance

jax.config.update("jax_enable_x64", True)

def print_stats(arr):
    print("mean", np.mean(arr))
    print("std", np.std(arr))
    print("median", np.median(arr))
    print("Q1", np.percentile(arr, 25))
    print("Q3", np.percentile(arr, 75))
    print("IQR/2", (np.percentile(arr, 75) - np.percentile(arr, 25))/2)

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
def do_test(thetas, x_vec_0_test, traj_test, X_train, Y_train):
    X_test = traj_test[:-1]
    Y_test = traj_test[1:]

    N_steps = len(X_test)
    
    HD_losses = []
    deviate_steps = []
    mse_losses = []
    
    for theta in thetas:
        pred_func = get_pred_funcs(theta, ext_kernel, X_train, Y_train)
        
        traj_pred = predict_series(pred_func, jnp.array(x_vec_0_test), N_steps)
        
        curr_HD = max(hausdorff_distance(traj_test, traj_pred), hausdorff_distance(traj_pred, traj_test))
        HD_losses.append(curr_HD)
        
        curr_deviate_steps = []
        for eps in [0.1, 0.25]:
            curr_deviate_steps.append(np.argmin(np.linalg.norm(traj_test - traj_pred, axis=1) / np.linalg.norm(traj_test, axis=1) < eps))
        deviate_steps.append(curr_deviate_steps)
        
        curr_mse_loss = get_NA_loss(pred_func, X_test, Y_test)
        mse_losses.append(curr_mse_loss)
    
    return HD_losses, deviate_steps, mse_losses
        
        
# ======== Latex utils =========


def format_number_without_exponent(number, exponent, significant_figures=3):
    # Scale the number by 10 to the power of the negative exponent
    scaled_number = number * (10 ** (-1 * exponent))
    # Format the scaled number to the desired number of significant figures
    formatted_number = f"{scaled_number:.{significant_figures - 1}f}"
    return formatted_number
                          

def latex_stats_arr(arr):
    median_val = np.median(arr)
    significant_figures = 3
    
    formatted_number = f"{median_val:.{significant_figures - 1}e}"

    median_base, exponent = formatted_number.split("e")
#     f"{base} \\times 10^{{{int(exponent)}}}"
    
    exponent = int(exponent)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    stat = ""
    stat += f"{median_base}"
    
    if q1 != q3:                 
        stat += f" \\ [{format_number_without_exponent(q1, exponent)}, {format_number_without_exponent(q3, exponent)}]"
    if exponent == 0:
        return stat
    return f"{stat} \\times 10^{{{exponent}}}"
                          
                          

def get_latex_test(res):

    text = ""

    text += "MSE"
    text += " & "
    text += f"${latex_stats_arr(res[f'mse_losses_sos'])}$"  
    text += " & "
    text += f"${latex_stats_arr(res[f'mse_losses_flow'])}$"  
    text += " \\\\ "

    text += "HD"
    text += " & "
    text += f"${latex_stats_arr(res[f'HD_losses_sos'])}$"
    text += " & "
    text += f"${latex_stats_arr(res[f'HD_losses_flow'])}$"
    text += " \\\\ "

    text += "Deviation (0.1)"
    text += " & "
    text += f"${latex_stats_arr(np.array(res[f'deviate_steps_sos'])[:, 0])}$"
    text += " & "
    text += f"${latex_stats_arr(np.array(res[f'deviate_steps_flow'])[:, 0])}$"
    text += " \\\\ "

    text += "Deviation (0.25)"
    text += " & "
    text += f"${latex_stats_arr(np.array(res[f'deviate_steps_sos'])[:, 1])}$"
    text += " & "
    text += f"${latex_stats_arr(np.array(res[f'deviate_steps_flow'])[:, 1])}$"
    text += " \\\\ "
    print(text)

def get_latex_train(res):

    text = ""

    text += "MSE"
    text += " & "
    text += f"${latex_stats_arr(res[f'mse_losses_sos_train'])}$"  
    text += " & "
    text += f"${latex_stats_arr(res[f'mse_losses_flow_train'])}$"  
    text += " \\\\ "

    text += "HD"
    text += " & "
    text += f"${latex_stats_arr(res[f'HD_losses_sos_train'])}$"
    text += " & "
    text += f"${latex_stats_arr(res[f'HD_losses_flow_train'])}$"
    text += " \\\\ "

    text += "Deviation (0.1)"
    text += " & "
    text += f"${latex_stats_arr(np.array(res[f'deviate_steps_sos_train'])[:, 0])}$"
    text += " & "
    text += f"${latex_stats_arr(np.array(res[f'deviate_steps_flow_train'])[:, 0])}$"
    text += " \\\\ "

    text += "Deviation (0.25)"
    text += " & "
    text += f"${latex_stats_arr(np.array(res[f'deviate_steps_sos_train'])[:, 1])}$"
    text += " & "
    text += f"${latex_stats_arr(np.array(res[f'deviate_steps_flow_train'])[:, 1])}$"
    text += " \\\\ "
    print(text)


def get_latex_rho(res):

    text = ""

    text += "Rho"
    text += " & "
    text += f"${latex_stats_arr(res[f'losses_sos'])}$"  
    text += " & "
    text += f"${latex_stats_arr(res[f'losses_flow'])}$"  
    text += " \\\\ "

    print(text)
