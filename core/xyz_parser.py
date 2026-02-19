import numpy as np

def parse_xyz(file):

    lines = file.read().decode().splitlines()[2:]

    elements = []
    coords = []

    for line in lines:
        e, x, y, z = line.split()
        elements.append(e)
        coords.append([float(x), float(y), float(z)])

    return elements, np.array(coords)
