import numpy as np
import pandas as pd

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

def mass_dependent(model_name,base_path = base_path):

    # Read mass bin
    rad_m = pd.read_csv(base_path + model_name + "/mass/mass_0.0-0.5_mass_bin.dat", delimiter=" ", header=None)#.dropna(axis=1)
    m = rad_m.to_numpy()  # shape: (N_rows, varying columns)
    
    # Read sigma_phi
    disp_phi_m = pd.read_csv(base_path + model_name +"/mass/mass_0.0-0.5_disp_phi.dat", delimiter=" ", header=None)#.dropna(axis=1)
    s_phi_m = disp_phi_m.to_numpy()

     # Read sigma_phi
    vel_phi_m = pd.read_csv(base_path + model_name +"/mass/mass_0.0-0.5_vphi.dat", delimiter=" ", header=None)#.dropna(axis=1)
    v_phi_m = vel_phi_m.to_numpy()
    
    # Read age
    age = pd.read_csv(base_path + model_name + "/age.dat",
                      sep=r"\s+", header=None).to_numpy().flatten()  # shape (N_rows,)

    
    return {"name": model_name,
            "age": age,
            "m": m,
            "disp_phi_m": s_phi_m,
            "v_phi_m": v_phi_m
            }

def select_snapshots(model,n=100):
    s_phi_m = model["s_phi_m"]
    v_phi_m = model["v_phi_m"]
    m = model["m"]
    age = model["age"]

    selected_indices = range(0, len(m), n)
    m_all = np.concatenate([m[idx] for idx in selected_indices])
    s_phi_m_all = np.concatenate([s_phi_m[idx] for idx in selected_indices])
    v_phi_m_all = np.concatenate([v_phi_m[idx] for idx in selected_indices])
    age_all = np.concatenate([np.full_like(m[idx], age[idx]) for idx in selected_indices])

    return {"name": model["name"],
            "n_snapshots": n,
            "age": age_all,
            "m": m_all,
            "s_phi_m": s_phi_m_all,
            "v_phi_m": v_phi_m_all}