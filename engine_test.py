from __future__ import print_function
# %matplotlib inline
import matplotlib.pyplot as plt
import openpathsampling as paths

import openpathsampling.engines.openmm as omm
from simtk.openmm import app
import simtk.openmm as mm
import simtk.unit as unit
from openmmtools.integrators import VVVRIntegrator


import mdtraj as md

import numpy as np

import os

# initial_pdb_ase = read(filename="AD_initial_frame.pdb")

initial_pdb = os.path.join("AD_initial_frame.pdb")
forcefield = app.ForceField('amber96.xml', 'tip3p.xml')
pdb = app.PDBFile(initial_pdb)
system = forcefield.createSystem(
    pdb.topology,
    nonbondedMethod=app.PME,
    nonbondedCutoff=1.0*unit.nanometers,
    constraints=app.HBonds,
    rigidWater=True,
    ewaldErrorTolerance=0.0005
)
hi_T_integrator = VVVRIntegrator(
    500*unit.kelvin,
    1.0/unit.picoseconds,
    2.0*unit.femtoseconds)
hi_T_integrator.setConstraintTolerance(0.00001)


template = omm.snapshot_from_pdb(initial_pdb)
openmm_properties = {'OpenCLPrecision': 'mixed'}
engine_options = {
    'n_steps_per_frame': 10,
    'n_frames_max': 2000
}

hi_T_engine = omm.Engine(
    template.topology,
    system,
    hi_T_integrator,
    openmm_properties=openmm_properties,
    options=engine_options
)
hi_T_engine.name = '500K'

hi_T_engine.current_snapshot = template
# hi_T_engine.minimize()


# define the CVs
psi = paths.MDTrajFunctionCV(
    "psi", md.compute_dihedrals,
    topology=template.topology, indices=[[6,8,14,16]]
).enable_diskcache()
phi = paths.MDTrajFunctionCV(
    "phi", md.compute_dihedrals,
    topology=template.topology, indices=[[4,6,8,14]]
).enable_diskcache()



# define the states
deg = 180.0/np.pi
C_7eq = (
    paths.PeriodicCVDefinedVolume(phi, lambda_min=-180/deg, lambda_max=0/deg,
                                  period_min=-np.pi, period_max=np.pi) &
    paths.PeriodicCVDefinedVolume(psi, lambda_min=100/deg, lambda_max=200/deg,
                                  period_min=-np.pi, period_max=np.pi)
).named("C_7eq")
# similarly, without bothering with the labels:
alpha_R = (
    paths.PeriodicCVDefinedVolume(phi, -180/deg, 0/deg, -np.pi, np.pi) &
    paths.PeriodicCVDefinedVolume(psi, -100/deg, 0/deg, -np.pi, np.pi)
).named("alpha_R")

# assert f(n_steps, found_vol, all_vol, tstep) == \
#         "Ran 100 frames [50.0]. Found states [A,B]. Looking for [C,D].\n"


visit_all = paths.VisitAllStatesEnsemble(states=[C_7eq, alpha_R])

# all_states=visit_all.states
# for state in all_states:
#     print(state)
#
# list_of_states=len([visit_all.can_append])
# print(list_of_states)

trajectory = hi_T_engine.generate(hi_T_engine.current_snapshot, [visit_all.can_append])

tmp_network = paths.TPSNetwork.from_states_all_to_all([C_7eq, alpha_R])


# take the subtrajectory matching the ensemble (only one ensemble, only one subtraj)
subtrajectories = []
for ens in tmp_network.analysis_ensembles:
    subtrajectories += ens.split(trajectory)
print(subtrajectories)

plt.plot(phi(trajectory), psi(trajectory), 'k.')
plt.plot(phi(subtrajectories[0]), psi(subtrajectories[0]), 'r')
plt.show()


