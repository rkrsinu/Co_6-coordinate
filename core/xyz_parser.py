import numpy as np

# Full periodic table (Z → symbol)
Z_TO_ELEMENT = {
    1:"H",2:"He",3:"Li",4:"Be",5:"B",6:"C",7:"N",8:"O",9:"F",10:"Ne",
    11:"Na",12:"Mg",13:"Al",14:"Si",15:"P",16:"S",17:"Cl",18:"Ar",
    19:"K",20:"Ca",21:"Sc",22:"Ti",23:"V",24:"Cr",25:"Mn",26:"Fe",27:"Co",28:"Ni",29:"Cu",30:"Zn",
    31:"Ga",32:"Ge",33:"As",34:"Se",35:"Br",36:"Kr",
    37:"Rb",38:"Sr",39:"Y",40:"Zr",41:"Nb",42:"Mo",43:"Tc",44:"Ru",45:"Rh",46:"Pd",47:"Ag",48:"Cd",
    49:"In",50:"Sn",51:"Sb",52:"Te",53:"I",54:"Xe",
    55:"Cs",56:"Ba",57:"La",58:"Ce",59:"Pr",60:"Nd",61:"Pm",62:"Sm",63:"Eu",64:"Gd",65:"Tb",66:"Dy",
    67:"Ho",68:"Er",69:"Tm",70:"Yb",71:"Lu",
    72:"Hf",73:"Ta",74:"W",75:"Re",76:"Os",77:"Ir",78:"Pt",79:"Au",80:"Hg",
    81:"Tl",82:"Pb",83:"Bi",84:"Po",85:"At",86:"Rn"
}

PERIODIC_TABLE = set(Z_TO_ELEMENT.values())


def parse_xyz(uploaded_file):

    lines = uploaded_file.read().decode().splitlines()

    elements = []
    coords = []

    for line in lines:

        parts = line.split()
        if len(parts) < 4:
            continue

        atom = parts[0]

        # -------------------------
        # Case 1: atomic number
        # -------------------------
        if atom.isdigit():
            Z = int(atom)
            if Z not in Z_TO_ELEMENT:
                raise ValueError(f"Atomic number {Z} is not supported.")
            element = Z_TO_ELEMENT[Z]

        # -------------------------
        # Case 2: element symbol
        # -------------------------
        else:
            element = atom.capitalize()
            if element not in PERIODIC_TABLE:
                raise ValueError(f"Unknown element '{element}'")

        # -------------------------
        # Coordinates
        # -------------------------
        try:
            x, y, z = map(float, parts[1:4])
        except ValueError:
            continue

        elements.append(element)
        coords.append([x, y, z])

    if len(elements) == 0:
        raise ValueError("No valid atomic coordinates found.")

    # 🔥 Important: your model requires Co center
    if "Co" not in elements:
        raise ValueError("Only Co-containing complexes are supported.")

    return elements, np.array(coords, dtype=float)
