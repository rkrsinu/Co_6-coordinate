import numpy as np

def build_feature_vector(bond_lengths, bond_angles, ideal_dev):

    features = bond_lengths + bond_angles + [ideal_dev]

    return np.array(features, dtype=float)
