from __future__ import print_function
from typing import List
from pathlib import Path
import re

import numpy as np
from ase.io import read, write
from ase.md import VelocityVerlet, MDLogger, Langevin
from ase.calculators.lj import LennardJones
import ase.units as units
from ase.build import molecule

import sys

sys.path.append("..")
from ops_ase.engine import AseEngine
from ops_ase.topology import AseTopology as Topology
import openpathsampling as ops

import openpathsampling.engines.openmm as ops_openmm
# %matplotlib inline
import matplotlib.pyplot as plt
import simtk.unit as unit
from openmmtools.integrators import VVVRIntegrator
import os

# initial_pdb = os.path.join("AD_initial_frame.pdb")

atoms = read(filename=Path(__file__).parent.parent / "original_tutorial" / "AD_initial_frame_without_water.pdb")
atoms.calc = LennardJones()

hi_T_integrator = Langevin(atoms=atoms, timestep=2 * units.fs, temperature_K=500, friction=0.01, logfile='./md.log')
deg = 180.0 / np.pi


def get_engine():
    ops_topology = ops_openmm.tools.topology_from_pdb("../ops_ase_tutorial/AD_initial_frame_without_water.pdb")
    engine_options = {
        'n_steps_per_frame': 10,
        'n_frames_max': 6000
    }
    hi_T_engine = AseEngine(
        ops_topology,
        hi_T_integrator,
        options=engine_options,
    )
    # current_snapshot = hi_T_engine.current_snapshot
    return hi_T_engine


# current_snapshot = hi_T_engine.current_snapshot
# hi_T_engine.minimize()

def wrap_pi_to_pi(x):
    return x - 2 * np.pi if x > np.pi else x


def f(atoms, indices):
    radian = atoms.get_dihedrals(indices) / deg
    return wrap_pi_to_pi(radian)


def get_states():
    psi = ops.FunctionCV(name="psi",
                         f=f,
                         indices=[[6, 8, 14, 16]],
                         cv_requires_lists=False,
                         cv_wrap_numpy_array=True,
                         cv_scalarize_numpy_singletons=True
                         ).enable_diskcache()
    phi = ops.FunctionCV(name="phi",
                         f=f,
                         indices=[[4, 6, 8, 14]],
                         cv_requires_lists=False,
                         cv_wrap_numpy_array=True,
                         cv_scalarize_numpy_singletons=True
                         ).enable_diskcache()

    # define the states

    C_7eq = (
            ops.PeriodicCVDefinedVolume(phi, lambda_min=-180 / deg, lambda_max=0 / deg,
                                        period_min=-np.pi, period_max=np.pi) &
            ops.PeriodicCVDefinedVolume(psi, lambda_min=100 / deg, lambda_max=200 / deg,
                                        period_min=-np.pi, period_max=np.pi)
    ).named("C_7eq")
    # similarly, without bothering with the labels:
    alpha_R = (
            ops.PeriodicCVDefinedVolume(phi, -180 / deg, 0 / deg, -np.pi, np.pi) &
            ops.PeriodicCVDefinedVolume(psi, -100 / deg, 0 / deg, -np.pi, np.pi)
    ).named("alpha_R")
    return C_7eq, alpha_R


# the VisitAllStatesEnsemble and engine generate are to create a high-temperature trajectory that visits all of our states.
# Here we only have 2 states, but this approach generalizes to multiple states.

def get_trajectory():
    engine = get_engine()
    states = get_states()
    current_snapshot = engine.current_snapshot
    # visit_all = ops.VisitAllStatesEnsemble([C_7eq, alpha_R])
    # trajectory = hi_T_engine.generate(snapshot=hi_T_engine.current_snapshot, running=[visit_all.can_append])
    visit_all = ops.VisitAllStatesEnsemble([states])
    trajectory = engine.generate(snapshot=engine.current_snapshot, running=[visit_all.can_append])

    # atoms_obj=trajectory[1].integrator.atoms

    for i in range(trajectory.n_snapshots):
        atoms = trajectory[i].integrator.atoms
        lattic = atoms.cell.array
        symbol = atoms.symbols
        symbol_count = [1 for i in range(len(list(symbol)))]
        system = 'AD_withoutwater'
        systemsize = "1.00000"
        Cartesian = 'Cartesian'
        position = atoms.positions

        s = (system + '\n')
        s += (systemsize + '\n')

        s += (re.sub('[\[\]]', '', np.array_str(lattic, precision=4)) + '\n')

        s += (" ".join(list(symbol)) + '\n')
        s += (" ".join([str(s) for s in symbol_count]) + '\n')
        s += (Cartesian + '\n')

        s += (re.sub('[\[\]]', '', np.array_str(position, precision=4)) + '\n')

        with open(f"POSCAR_{i}.vasp", 'w') as file:
            file.write(s)


if __name__ == "__main__":
    get_trajectory()
# # # create a network so we can use its ensemble to obtain an initial trajectory
# # # use all-to-all because initial traj can be A->B or B->A; will be reversed
# tmp_network = ops.TPSNetwork.from_states_all_to_all([C_7eq, alpha_R])
#
# init_traj_storage = ops.Storage("initial_trajectory.nc", 'w', template=current_snapshot)
# init_traj_storage.save(trajectory)
#
# #
# # print(init_traj_storage.engines)
# # print(init_traj_storage.snapshots[3])
# # take the subtrajectory matching the ensemble (only one ensemble, only one subtraj)
# subtrajectories = []
# for ens in tmp_network.analysis_ensembles:
#     subtrajectories += ens.split(trajectory)
# print(subtrajectories)
#
# plt.plot(phi(trajectory)  * deg, psi(trajectory) * deg, 'k.')
# plt.plot(phi(subtrajectories[0]) * deg, psi(subtrajectories[0])  * deg, 'r')
# plt.xlabel('phi', fontsize=18)
# plt.ylabel('psi', fontsize=18)
#
# plt.show()