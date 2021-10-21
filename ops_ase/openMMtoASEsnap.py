import torch
from simtk.unit.unit_definitions import kilocalories_per_mole

import openpathsampling as ops
import openpathsampling.engines.snapshot as ops_snapshot
import openpathsampling.engines.openmm.snapshot as openmm_snapshot
import ase

import os
import sys
# sys.path.append(os.path.abspath('../'))
sys.path.append("/Users/a358100/Documents/codes/ops_ase")
# import ops_ase.engine as ase_engine
import ops_ase.engine as ase_engine

# from .. import ops_ase.engine as ase_energy
from ase.io import read
from ase.md import VelocityVerlet, MDLogger, Langevin
import ase.units as units
from ase.io.netcdftrajectory import read_netcdftrajectory,NetCDFTrajectory
from pathlib import Path
from mdtraj.core.element import Element

from hippynn.interfaces.ase_interface import HippynnCalculator

class OpenmmSnapshotToAseSnapshot:
    @classmethod
    def generate_ase_snapshot(cls, snapshot: openmm_snapshot.Snapshot) -> ase_engine.AseSnapshot:
        ase_atoms = cls.generate_atoms(snapshot=snapshot)
        ase_snapshot = ase_engine.AseSnapshot(atoms=ase_atoms)
        return ase_snapshot

    @staticmethod
    def generate_ase_atoms(snapshot: openmm_snapshot.Snapshot) -> ase.atoms.Atoms:
        symbols = [atom.element.symbol for atom in snapshot.topology.mdtraj.atoms]
        masses = [atom.element.mass for atom in snapshot.topology.mdtraj.atoms]  # atomic mass of the element
        radiuses = [atom.element.radius for atom in snapshot.topology.mdtraj.atoms] # van der Waals radius in nm

        ase_atoms = []
        for i in range(snapshot.topology.n_atoms):
            ase_atom = ase.atom.Atom(symbol=symbols[i],
                                     position=snapshot.coordinates[i]*10,
                                     tag=None,
                                    #  momentum=snapshot.velocities[i] *1000 * masses[i], # ToDo: check the ASE momentum unit, openmm velocity nm/ps
                                     mass=masses[i], # in atomic unit
                                     magmom=None, # ToDo: handle magmom cases
                                     charge=None # Todo: figure out charge
                                     )
            ase_atoms.append(ase_atom)

        ase_atoms = ase.atoms.Atoms(symbols=ase_atoms,
                                    cell=snapshot.box_vectors._value*10,  # ToDo: current hard-coded in angstrom
                                    pbc=None,
                                    # positions=snapshot.coordinates*10,
                                    )
        return ase_atoms