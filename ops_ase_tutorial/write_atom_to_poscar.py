
from pathlib import Path
import re
import numpy as np
from ase.io import read, write
from ase.md import VelocityVerlet, MDLogger, Langevin
from ase.calculators.lj import LennardJones
import ase.units as units

atoms = read(filename=Path(__file__).parent.parent/"original_tutorial"/"AD_initial_frame_without_water.pdb")
atoms.calc = LennardJones()

integrator = Langevin(atoms=atoms, timestep=2*units.fs, temperature_K=500, friction=0.01, logfile='./md.log')
atoms_obj=integrator.atoms
lattic = atoms.cell.array
symbol = atoms.symbols
symbol_count=[1 for i in range(len(list(symbol)))]
system= 'AD_withoutwater'
systemsize= "1.00000"
Cartesian= 'Cartesian'
position = atoms.positions

s = (system + '\n')
s += (systemsize + '\n')

s += (re.sub('[\[\]]', '', np.array_str(lattic, precision=4)) + '\n')

s += (" ".join(list(symbol)) + '\n')
s += (" ".join([str(s) for s in symbol_count]) + '\n')
s += (Cartesian + '\n')

s += (re.sub('[\[\]]', '', np.array_str(position, precision=4)) + '\n')

with open("POSCAR.vasp", 'w') as file:
    file.write(s)


