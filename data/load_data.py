# External packages
import os
import numpy as np
import pandas as pd
import copy

DATA_PATH = f"/pbs/throng/training/astroinfo2025/data/Nbody/"

def list_folders(path):
    # List all entries in the directory given by path
    entries = os.listdir(path)
    # Filter out only directories
    folders = [entry for entry in entries if os.path.isdir(os.path.join(path, entry))]
    return list(filter(lambda item: item != 'README', folders))

def load_data(model_name, myvars, simrange='mass_0.0-0.5', simtype='mass'):

    data = {}
    for name, filetag in myvars.items():
        data[name] = pd.read_csv(DATA_PATH + model_name + "/{}/{}_{}.dat".format(simtype,simrange,filetag), \
                                 delimiter=" ", header=None).to_numpy()
    data['age'] = pd.read_csv(DATA_PATH + model_name + "/age.dat", sep=r"\s+", header=None).to_numpy().flatten()
    
    return data

def dropna(data):
    selected_idx = ~np.isnan(data['mass'])
    for name in set(data.keys()) - set(['age']):
        data[name] = data[name][selected_idx]
    return data


def mass_dependent(model_name,base_path = DATA_PATH):
    """
    Load a dataset containing mass dependent quantities (s_phi/v_phi).

    Parameters
    ----------
        model_name: str, name of the model/simulation you want to load
        base_path: str, where to find the data

    Return
    ----------
        model_name: self explanatory
        age: time steps
        m: mass bins/time step
        s_phi_m: dispersion in function of mass bin/time step
        v_phi_m: velocity in function of mass bin/time step
    """
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
            "s_phi_m": s_phi_m,
            "v_phi_m": v_phi_m
            }

def remove_nan(model,var_name = "s_phi_m"):
    """
    Create a dataset without NaN for a specific quantity (default s_phi_m)

    Parameters
    ----------
        model: model returned by mass_dependant()
        var_name: str, variable wrt which you want to remove the NaNs (s_phi_m or v_phi_m)

    Returns
    ----------
        ds: Dictionnary with model_name, age, mass, var_name similar to mass_dependent()
    """
    var = model[var_name]
    m = model["m"]
    age = model["age"]
    ds = {"name":model["name"],"age":[],var_name:[], "m":[]}

    for i in range(var.shape[0]):
        mask = np.isnan(var[i])
        if len(var[i][~mask]) >0:
            assert len(var[i][~mask]) == len(m[i][~mask])
            tmp_var = var[i][~mask]
            tmp_m = m[i][~mask]
            ds[var_name].append(tmp_var)
            ds["age"].append(age[i])
            ds["m"].append(tmp_m.reshape(1,len(tmp_m)))

    return ds



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