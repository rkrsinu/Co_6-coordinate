import numpy as np

def build_features(bl, angles, ideal_dev):

    bl = np.array(bl, dtype=float)
    angles = np.array(angles, dtype=float)

    avg_d = np.mean(bl)
    var_d = np.var(bl)
    var_A = np.var(angles)

    # axial / equatorial split
    Req = np.mean(bl[:4])
    Rax = np.mean(bl[4:])
    Delta_R = Rax - Req

    features = list(bl) + list(angles) + [
        ideal_dev,
        avg_d,
        var_d,
        var_A,
        Req,
        Rax,
        Delta_R
    ]

    if len(features) != 28:
        raise ValueError(f"Feature vector must have 28 values. Got {len(features)}")

    # 🔥 FIX: use float64 (not float32)
    return np.array(features, dtype=np.float64)
