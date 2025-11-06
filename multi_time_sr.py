import numpy as np
import physo
import physo.learn.monitoring as monitoring
from data import load_data
import os 
import multiprocessing
import torch
import sys

sys.path.append('/pbs/home/a/astropart22/.local/lib/python3.11/site-packages')
#sys.path.append('/pbs/home/a/astropart22/.local/lib/python3.9/site-packages')
import datetime

# Control thread and process settings
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)

# Set the start method for multiprocessing
multiprocessing.set_start_method('fork', force=True)

models=['1.5M_A_R4_10',
'500k_A_R2_10',
'500k_A_R4_10',
'500k_C_R4_10',
'250k_A_R2_25',
'250k_A_R2_25_vlk',
'250k_A_R2_10',
'250k_A_R2_5',
'250k_A_R4_25',
'250k_A_R4_25_imf50',
'250k_A_R4_25_lk',
'250k_A_R4_25_retr',
'250k_A_R4_25_vlk',
'250k_A_R4_10',
'250k_A_R4_10_retr',
'250k_B_R4_25',
'250k_B_R4_25_lk',
'250k_C_R2_10',
'250k_C_R4_25',
'250k_C_R4_25_lk',
'250k_C_R4_10',
'250k_W6_R4_25',
'250k_W6_R4_25_retr',
'500k_A_R4_LC_part1',
'500k_A_R4_LC_part2']

base_path = f"/pbs/throng/training/astroinfo2025/data/Nbody/"

# loading model
model1 = load_data.mass_dependent(model_name ='1.5M_A_R4_10',base_path = base_path)
s_phi_ds = load_data.remove_nan(model1, var_name= "s_phi_m",remove_massive=True)

def run_sr_analysis(s_phi_multiple, i):
    # Get the number of available CPUs from SLURM environment if available
    n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    if n_cpus == 1:
        n_cpus = int(os.environ.get('SLURM_NTASKS', 8))
    
    print(f"Running SR with {n_cpus} CPUs")

    save_path_training_curves = f'results/multi_time_1{i}.png'
    save_path_log             = f'results/multi_time_1{i}.log'
    
    run_logger     = lambda : monitoring.RunLogger(save_path = save_path_log,
                                                    do_save = True)
    
    run_visualiser = lambda : monitoring.RunVisualiser (epoch_refresh_rate = 1,
                                               save_path = save_path_training_curves,
                                               do_show   = False,
                                               do_prints = True,
                                               do_save   = True, )
    
    # Running SR task
    expression, logs = physo.ClassSR(s_phi_multiple["x"], s_phi_multiple["s_phi_m"],
                                # Giving names of variables (for display purposes)
                                X_names = [ "m" , "t"],
                                # Associated physical units (ignore or pass zeroes if irrelevant)
                                X_units = [ [0, 0, 1], [0, 1, 0]],
                                # Giving name of root variable (for display purposes)
                                y_name  = "s_phi",
                                y_units = [1, -1, 0],
                                # Fixed constants
                                fixed_consts       = [ 1.      ],
                                fixed_consts_units = [ [0, 0, 0] ],
                                # Whole class free constants
                                class_free_consts_names = [ "c0" ],
                                class_free_consts_units = [[0, 0, 0]],
                                # Realization specific free constants
                                spe_free_consts_names = [ "s0", "m0", "q", "t0"],
                                spe_free_consts_units = [ [1, -1, 0] , [0, 0, 1], [0,0,0], [0, 1, 0] ],
                                # Run config
                                run_config = physo.config.config1b.config1b,
                                # Symbolic operations that can be used to make f
                                op_names = ["mul", "add", "sub", "div", "inv", "n2", "sqrt", "neg", "pow"],
                                get_run_logger     = run_logger,
                                get_run_visualiser = run_visualiser,
                                # Parallel mode (only available when running from python scripts, not notebooks)
                                parallel_mode = True,
                                # Number of iterations
                                epochs = 1,
                                n_cpus=n_cpus)
    
    np.savetxt(f"results/x_{i}.dat",s_phi_multiple["x"])
    np.savetxt(f"results/x_{i}.dat",s_phi_multiple["s_phi_m"])
    return expression, logs

for i in range(1):
    s_phi_multiple = load_data.select_multiple_timesteps(s_phi_ds,n=10,q=3)
    ex,logs = run_sr_analysis(s_phi_multiple,i)
    print('-----------------------')