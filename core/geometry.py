import numpy as np
from itertools import combinations

def get_co_index(elements):
    for i, e in enumerate(elements):
        if e.upper() == "CO":
            return i
    raise ValueError("Co not found")

def get_6_neighbors(elements, coords, co_idx):
    co = coords[co_idx]

    dists = []
    for i in range(len(coords)):
        if i == co_idx:
            continue
        d = np.linalg.norm(coords[i] - co)
        dists.append((i, d))

    dists.sort(key=lambda x: x[1])

    neighbors = [i for i, _ in dists[:6]]
    return neighbors


def calc_bond_lengths(coords, co_idx, neighbors):
    co = coords[co_idx]
    return [np.linalg.norm(coords[i] - co) for i in neighbors]


def calc_angles(coords, co_idx, neighbors):
    co = coords[co_idx]
    angles = []

    for i, j in combinations(neighbors, 2):
        v1 = coords[i] - co
        v2 = coords[j] - co

        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))

        angles.append(ang)

    return angles[:15]
