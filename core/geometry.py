import numpy as np
from itertools import combinations


def find_co_index(elements):

    for i, e in enumerate(elements):
        if e.strip().lower().startswith("co"):
            return i

    return None


def angle(a, b, c):

    ba = a - b
    bc = c - b

    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))


def ideal_dev(angles):

    ideal = [90, 180]
    return float(np.mean([min(abs(a - i) for i in ideal) for a in angles]))


def compute_geometry(elements, coords):

    co_idx = find_co_index(elements)

    if co_idx is None:
        raise ValueError("No cobalt (Co) atom found in XYZ file.")

    co = coords[co_idx]

    dists = []

    for i in range(len(coords)):
        if i == co_idx:
            continue
        d = np.linalg.norm(coords[i] - co)
        dists.append((i, d))

    dists.sort(key=lambda x: x[1])

    if len(dists) < 6:
        raise ValueError("Less than 6 coordinating atoms found.")

    neighbors = [i for i, _ in dists[:6]]

    # bond lengths
    bond_lengths = [np.linalg.norm(coords[i] - co) for i in neighbors]

    # angles
    bond_angles = []

    for i, j in combinations(neighbors, 2):
        bond_angles.append(angle(coords[i], co, coords[j]))

    bond_angles = bond_angles[:15]

    if len(bond_angles) != 15:
        raise ValueError("Could not compute 15 bond angles.")

    return bond_lengths, bond_angles, ideal_dev(bond_angles)
