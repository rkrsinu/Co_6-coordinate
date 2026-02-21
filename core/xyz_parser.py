import numpy as np

# Allowed elements for this app
Z_TO_ELEMENT = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    14: "Si",   # NEW
    15: "P",
    16: "S",
    17: "Cl",
    27: "Co",
    34: "Se",   # NEW
    35: "Br",   # NEW
    53: "I"     # NEW
}

ALLOWED_ELEMENTS = set(Z_TO_ELEMENT.values())


def parse_xyz(uploaded_file):

    lines = uploaded_file.read().decode().splitlines()

    elements = []
    coords = []

    for line in lines:

        parts = line.split()

        if len(parts) < 4:
            continue

        atom = parts[0]

        # ---------- atomic number ----------
        if atom.isdigit():

            Z = int(atom)

            if Z not in Z_TO_ELEMENT:
                raise ValueError(f"Atomic number {Z} is not supported in this app.")

            element = Z_TO_ELEMENT[Z]

        # ---------- element symbol ----------
        else:
            element = atom.capitalize()

            if element not in ALLOWED_ELEMENTS:
                raise ValueError(f"Element '{element}' is not supported in this app.")

        # ---------- coordinates ----------
        try:
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
        except ValueError:
            continue

        elements.append(element)
        coords.append([x, y, z])

    if len(elements) == 0:
        raise ValueError("No valid atomic coordinates found.")

    # ✅ enforce Co-only metal
    if "Co" not in elements:
        raise ValueError("Only Co-containing complexes are supported.")

    return elements, np.array(coords, dtype=float)
