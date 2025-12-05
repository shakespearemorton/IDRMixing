import openmm as mm
import openmm.app as app
import openmm.unit as unit

amino_acids = {
    1: 'MET',
    2: 'GLY',
    3: 'LYS',
    4: 'THR',
    5: 'ARG',
    6: 'ALA',
    7: 'ASP',
    8: 'GLU',
    9: 'TYR',
    10: 'VAL',
    11: 'LEU',
    12: 'GLN',
    13: 'TRP',
    14: 'PHE',
    15: 'SER',
    16: 'HIS',
    17: 'ASN',
    18: 'PRO',
    19: 'CYS',
    20: 'ILE'
}

# Map amino acid masses and sizes
mass = {
    'MET': 131.19, 'GLY': 57.05, 'LYS': 128.20, 'THR': 101.10, 'ARG': 156.20,
    'ALA': 71.08, 'ASP': 115.10, 'GLU': 129.10, 'TYR': 163.20, 'VAL': 99.07,
    'LEU': 113.20, 'GLN': 128.10, 'TRP': 186.20, 'PHE': 147.20, 'SER': 87.08,
    'HIS': 137.10, 'ASN': 114.10, 'PRO': 97.12, 'CYS': 103.10, 'ILE': 113.20
}

size = {
    'MET': 6.18, 'GLY': 4.5, 'LYS': 6.36, 'THR': 5.62, 'ARG': 6.56,
    'ALA': 5.04, 'ASP': 5.58, 'GLU': 5.92, 'TYR': 6.46, 'VAL': 5.86,
    'LEU': 6.18, 'GLN': 6.02, 'TRP': 6.78, 'PHE': 6.36, 'SER': 5.18,
    'HIS': 6.08, 'ASN': 5.68, 'PRO': 5.56, 'CYS': 5.48, 'ILE': 6.18
}

charge = {
    'MET': 0, 'GLY': 0, 'LYS': 1, 'THR': 0, 'ARG': 1,
    'ALA': 0, 'ASP': -1, 'GLU': -1, 'TYR': 0, 'VAL': 0,
    'LEU': 0, 'GLN': 0, 'TRP': 0, 'PHE': 0, 'SER': 0,
    'HIS': 0, 'ASN': 0, 'PRO': 0, 'CYS': 0, 'ILE': 0
}

hydropathy = {
    'MET': 0.5308481134337497, 'GLY': 0.7058843733666401, 'LYS': 0.1790211738990582, 'THR': 0.3713162976273964, 'ARG': 0.7307624767517166,
    'ALA': 0.2743297969040348, 'ASP': 0.0416040480605567, 'GLU': 0.0006935460962935, 'TYR': 0.9774611449343455, 'VAL': 0.2083769608174481,
    'LEU': 0.6440005007782226, 'GLN': 0.3934318551056041, 'TRP': 0.9893764740371644, 'PHE': 0.8672358982062975, 'SER': 0.4625416811611541,
    'HIS': 0.4663667290557992, 'ASN': 0.4255859009787713, 'PRO': 0.3593126576364644, 'CYS': 0.5615435099141777, 'ILE': 0.5423623610671892
}

single_letter = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

reverse = {v: k for k, v in amino_acids.items()}
reverse_single = {v: k for k, v in single_letter.items()}
_kcal_to_kj = 4.184
epsilon=0.2*_kcal_to_kj
mu=1,
delta=0.08
use_pbc = True
ldby=1*unit.nanometer
dielectric_water=80.0
NA = unit.AVOGADRO_CONSTANT_NA # Avogadro constant
kB = unit.BOLTZMANN_CONSTANT_kB  # Boltzmann constant
EC = 1.602176634e-19*unit.coulomb # elementary charge
VEP = 8.8541878128e-12*unit.farad/unit.meter # vacuum electric permittivity
GAS_CONST = 1.0*unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA # gas constant

