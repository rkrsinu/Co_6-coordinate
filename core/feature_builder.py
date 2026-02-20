import numpy as np

def build_features(bl, angles, ideal_dev):

    features = list(bl) + list(angles) + [ideal_dev]

    if len(features) != 22:
        raise ValueError("Feature vector must have 22 values.")

    return np.array(features, dtype=float)
