import numpy as np
from itertools import combinations

MIN_ANGLE = 60.0
CO_H_CUTOFF = 1.8


# ---------- helpers ----------
def find_co_index(elements):
    for i, e in enumerate(elements):
        if e.strip() == "Co":
            return i
    return None


def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))


def ideal_dev(angles):
    ideal = [90, 180]
    return float(np.mean([min(abs(a - i) for i in ideal) for a in angles]))


# ---------- robust octahedral neighbour finder ----------
def get_valid_octahedral_neighbors(elements, coords, co_index):

    co_coord = coords[co_index]
    candidates = []

    for i, el in enumerate(elements):

        if i == co_index:
            continue

        d = np.linalg.norm(coords[i] - co_coord)

        # ignore distant H
        if el == "H" and d >= CO_H_CUTOFF:
            continue

        candidates.append((i, d))

    candidates.sort(key=lambda x: x[1])

    if len(candidates) < 6:
        return None

    selected = [c[0] for c in candidates[:6]]
    pool = [c[0] for c in candidates[6:]]

    # enforce octahedral angles
    while True:

        bad_pair = None

        for i, j in combinations(selected, 2):

            ang = calculate_angle(coords[i], co_coord, coords[j])

            if ang < MIN_ANGLE:
                bad_pair = (i, j)
                break

        if bad_pair is None:
            break

        i, j = bad_pair

        di = np.linalg.norm(coords[i] - co_coord)
        dj = np.linalg.norm(coords[j] - co_coord)

        remove_atom = i if di > dj else j
        selected.remove(remove_atom)

        if not pool:
            return None

        selected.append(pool.pop(0))

    return selected


# ---------- MAIN FUNCTION ----------
def compute_geometry(elements, coords):

    co_idx = find_co_index(elements)

    if co_idx is None:
        raise ValueError("No cobalt (Co) atom found.")

    neighbors = get_valid_octahedral_neighbors(elements, coords, co_idx)

    if neighbors is None:
        raise ValueError("Valid octahedral coordination not found.")

    co_coord = coords[co_idx]

    # ----- bond lengths (sorted) -----
    bond_lengths = sorted([
        np.linalg.norm(coords[i] - co_coord)
        for i in neighbors
    ])

    # ----- bond angles (sorted) -----
    angles = sorted([
        calculate_angle(coords[i], co_coord, coords[j])
        for i, j in combinations(neighbors, 2)
    ])

    if len(angles) != 15:
        raise ValueError("Could not compute 15 bond angles.")

    idev = ideal_dev(angles)

    return bond_lengths, angles, idev
