import numpy as np
from itertools import combinations

def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))


def ideal_geometry_deviation(angles):
    ideal = [90, 180]
    dev = [min(abs(a - i) for i in ideal) for a in angles]
    return float(np.mean(dev))


def get_octahedral_neighbors(elements, coords):

    co_idx = [i for i, e in enumerate(elements) if e.upper() == "CO"][0]
    co = coords[co_idx]

    dists = [(i, np.linalg.norm(coords[i] - co))
             for i in range(len(coords)) if i != co_idx]

    dists.sort(key=lambda x: x[1])

    neighbors = [i for i, _ in dists[:6]]

    return co_idx, neighbors


def compute_geometry(elements, coords):

    co_idx, neighbors = get_octahedral_neighbors(elements, coords)
    co = coords[co_idx]

    # bond lengths
    bond_lengths = [np.linalg.norm(coords[i] - co) for i in neighbors]

    # angles
    bond_angles = []
    for i, j in combinations(neighbors, 2):
        bond_angles.append(calculate_angle(coords[i], co, coords[j]))

    bond_angles = bond_angles[:15]

    # sorting (CRITICAL for your model)
    bond_lengths = sorted(bond_lengths)
    bond_angles = sorted(bond_angles)

    ideal_dev = ideal_geometry_deviation(bond_angles)

    return bond_lengths, bond_angles, ideal_dev
