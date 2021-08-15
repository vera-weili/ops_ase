from __future__ import print_function
from typing import List
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.md import VelocityVerlet, MDLogger, Langevin
import ase.units as units
from ase.build import molecule

import sys
sys.path.append("..")
from ops_tutorial.ops_ase_test.engine import AseEngine
from ops_tutorial.ops_ase_test.topology import AseTopology as Topology
import openpathsampling as ops


# %matplotlib inline
import matplotlib.pyplot as plt

import simtk.unit as unit
from openmmtools.integrators import VVVRIntegrator

import os
# initial_pdb = os.path.join("AD_initial_frame.pdb")
atoms = read(filename=Path(__file__).parent.parent/"AD_initial_frame.pdb")

hi_T_integrator = Langevin(atoms=atoms, timestep=1*units.fs, temperature_K=300, friction=0.01, logfile='./md.log')


# hi_T_integrator.run(10)
# hi_T_integrator.attach(MDLogger(hi_T_integrator, atoms_calc, 'md.log', header=False, stress=False, peratom=True, mode='a'), interval=1000)

topology= Topology(
    n_spatial = 2,
    masses =[1.0, 1.0]
)

engine_options = {
    'n_steps_per_frame': 10,
    'n_frames_max': 100
}
hi_T_engine = AseEngine(
    hi_T_integrator,
    options=engine_options,
    topology=topology
)
current_snapshot = hi_T_engine.current_snapshot
# hi_T_engine.minimize()

# define the CVs
# def func(atoms: Atoms, indices: List[List[int]]):
#     return atoms.get_dihedrals(indices)
#
#
# psi = ops.CoordinateFunctionCV(name="psi", f=func, indices=[[6, 8, 14, 16]]).enable_diskcache()
# phi = ops.CoordinateFunctionCV(name="phi", f=func, indices=[[4, 6, 8, 14]]).enable_diskcache()

# define the CVs

deg = 180.0/np.pi

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
def wrap_pi_to_pi(x):
    return x-2*np.pi if x > np.pi else x


def f(atoms, indices):
    radian=atoms.get_dihedrals(indices) / deg
    return wrap_pi_to_pi(radian)

print(f(atoms=atoms, indices=[[4, 6, 8, 14]]))

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
#


# define the states
deg = 180.0/np.pi
C_7eq = (
    ops.PeriodicCVDefinedVolume(phi, lambda_min=-180/deg, lambda_max=0/deg,
                                period_min=-np.pi, period_max=np.pi) &
    ops.PeriodicCVDefinedVolume(psi, lambda_min=100/deg, lambda_max=200/deg,
                                period_min=-np.pi, period_max=np.pi)
).named("C_7eq")
# similarly, without bothering with the labels:
alpha_R = (
    ops.PeriodicCVDefinedVolume(phi, -180/deg, 0/deg, -np.pi, np.pi) &
    ops.PeriodicCVDefinedVolume(psi, -100/deg, 0/deg, -np.pi, np.pi)
).named("alpha_R")


# the VisitAllStatesEnsemble and engine generate are to create a high-temperature trajectory that visits all of our states.
# Here we only have 2 states, but this approach generalizes to multiple states.
visit_all = ops.VisitAllStatesEnsemble([C_7eq, alpha_R])

trajectory = hi_T_engine.generate(snapshot=hi_T_engine.current_snapshot, running=[visit_all.can_append])

# # # create a network so we can use its ensemble to obtain an initial trajectory
# # # use all-to-all because initial traj can be A->B or B->A; will be reversed
tmp_network = ops.TPSNetwork.from_states_all_to_all([C_7eq, alpha_R])


# take the subtrajectory matching the ensemble (only one ensemble, only one subtraj)
subtrajectories = []
for ens in tmp_network.analysis_ensembles:
    subtrajectories += ens.split(trajectory)
print(subtrajectories)

plt.plot(phi(trajectory), psi(trajectory), 'k.')
plt.plot(phi(subtrajectories[0]), psi(subtrajectories[0]), 'r')
plt.show()
