import numpy as np

def build_features(bond_lengths, bond_angles):
    
    avg_d = np.mean(bond_lengths)
    var_d = np.var(bond_lengths)
    var_A = np.var(bond_angles)

    features = bond_lengths + bond_angles + [avg_d, var_d, var_A]

    return np.array(features)
