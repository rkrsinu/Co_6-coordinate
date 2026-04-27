import numpy as np

PERIODIC_TABLE = {
    "H","He","Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "In","Sn","Sb","Te","I","Xe",
    "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Tl","Pb","Bi","Po","At","Rn"
}

def parse_xyz(uploaded_file):

    lines = uploaded_file.read().decode().splitlines()

    elements = []
    coords = []

    for line in lines:

        parts = line.split()
        if len(parts) < 4:
            continue

        element = parts[0].capitalize()

        if element not in PERIODIC_TABLE:
            raise ValueError(f"Unknown element '{element}'")

        try:
            x, y, z = map(float, parts[1:4])
        except ValueError:
            continue

        elements.append(element)
        coords.append([x, y, z])

    if len(elements) == 0:
        raise ValueError("No valid atomic coordinates found.")

    if "Co" not in elements:
        raise ValueError("Only Co-containing complexes are supported.")

    return elements, np.array(coords, dtype=float)
