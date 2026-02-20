import numpy as np

def build_features(bond_lengths, bond_angles, ideal_dev):

    feature_vector = (
        list(bond_lengths) +
        list(bond_angles) +
        [ideal_dev]
    )

    return np.array(feature_vector, dtype=float)
