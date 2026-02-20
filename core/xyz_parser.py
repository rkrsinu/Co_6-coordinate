import numpy as np

def parse_xyz(uploaded_file):

    lines = uploaded_file.read().decode().splitlines()

    elements = []
    coords = []

    for line in lines:

        parts = line.split()

        # valid atom line must have 4 entries: Element x y z
        if len(parts) < 4:
            continue

        try:
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
        except ValueError:
            continue

        elements.append(parts[0])
        coords.append([x, y, z])

    if len(elements) == 0:
        raise ValueError("No atomic coordinates found in file.")

    return elements, np.array(coords, dtype=float)
