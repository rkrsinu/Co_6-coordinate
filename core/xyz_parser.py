import numpy as np

# Full periodic table symbols
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

        atom = parts[0]

        # Normalize symbol
        element = atom.capitalize()

        # ✅ Allow all periodic elements
        if element not in PERIODIC_TABLE:
            raise ValueError(f"Unknown element '{element}'")

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

    # 🔥 still enforce Co center (important for your model)
    if "Co" not in elements:
        raise ValueError("Only Co-containing complexes are supported.")

    return elements, np.array(coords, dtype=float)
