from __future__ import print_function
import sys
from pathlib import Path
sys.path.append("..")
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

from simtk.openmm import app
import simtk.openmm as mm
import simtk.unit as unit
import openmmtools

import openpathsampling as paths
import openpathsampling.engines.openmm as ops_openmm
import openpathsampling.engines.openmm as omm
import mdtraj as md


""""
1. Initial trajectory is read from a pre-defined trajectory in .nc file
2. Dihedrals and states are defined
3. Mover and schemes are defined
4. MC sampler is used
"""

deg = 180.0 / np.pi  # for conversion between radians and degrees


def get_engine(system, integrator):
    openmm_properties = {}
    engine_options = {
        'n_frames_max': 2000,
        'n_steps_per_frame': 10
    }
    ops_topology = ops_openmm.tools.topology_from_pdb("../ops_ase_tutorial/AD_initial_frame.pdb")
    engine = ops_openmm.Engine(
        topology=ops_topology,
        system=system,
        integrator=integrator,
        openmm_properties=openmm_properties,
        options=engine_options
    )
    engine.name = 'TPS MD Engine'
    return engine


def print_backbone(md_topology):
    for atom_number in md_topology.select("backbone"):
        print(atom_number, md_topology.atom(atom_number))
    """
    4 ACE1-C
    5 ACE1-O
    6 ALA2-N
    8 ALA2-CA
    14 ALA2-C
    15 ALA2-O
    16 NME3-N
    18 NME3-C
    """

def get_states(phi, psi):
    C_7eq = (
            paths.PeriodicCVDefinedVolume(phi, lambda_min=-180 / deg, lambda_max=0 / deg,
                                          period_min=-np.pi, period_max=np.pi)
            & paths.PeriodicCVDefinedVolume(psi, lambda_min=100 / deg, lambda_max=200 / deg,
                                            period_min=-np.pi, period_max=np.pi)
    ).named("C_7eq")
    alpha_R = (
            paths.PeriodicCVDefinedVolume(phi, -180 / deg, 0 / deg, -np.pi, np.pi) &
            paths.PeriodicCVDefinedVolume(psi, -100 / deg, 0 / deg, -np.pi, np.pi)
    ).named("alpha_R")
    return C_7eq, alpha_R

def run():
    # this cell is all OpenMM specific
    import os
    initial_pdb = os.path.join("AD_initial_frame.pdb")
    forcefield = app.ForceField('amber96.xml', 'tip3p.xml')
    system = forcefield.createSystem(
        topology=app.PDBFile("AD_initial_frame.pdb").topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometers,
        constraints=app.HBonds,
        rigidWater=True,
        ewaldErrorTolerance=0.0005
    )

    integrator = openmmtools.integrators.VVVRIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picoseconds,
        2.0 * unit.femtoseconds
    )
    integrator.setConstraintTolerance(0.00001)

    engine = get_engine(system, integrator)

    md_topology = md.Topology.from_openmm(engine.simulation.topology)
    print_backbone(md_topology)

    # ACE1-C, ALA2-N, ALA2-CA, ALA2-C: 4,6,8,14
    phi = paths.MDTrajFunctionCV("phi", md.compute_dihedrals, engine.topology, indices=[[4, 6, 8, 14]])
    psi = paths.MDTrajFunctionCV(name="psi",
                                 f=md.compute_dihedrals,
                                 topology=engine.topology,
                                 indices=[[6, 8, 14, 16]],
                                 )

    C_7eq, alpha_R = get_states(phi, psi)

    template = omm.snapshot_from_pdb(initial_pdb)
    init_traj_storage = paths.Storage("initial_trajectory.nc", 'w', template=template)

    visit_all = paths.VisitAllStatesEnsemble([C_7eq, alpha_R])

    trajectory = engine.generate(snapshot=engine.current_snapshot, running=[visit_all.can_append])

    # # # create a network so we can use its ensemble to obtain an initial trajectory
    # # # use all-to-all because initial traj can be A->B or B->A; will be reversed
    # tmp_network = paths.TPSNetwork.from_states_all_to_all([C_7eq, alpha_R])
    # print(init_traj_storage.list_stores())
    # init_traj = init_traj_storage.trajectories[0]
    #
    # plt.plot(phi(init_traj), psi(init_traj))
    # plt.xlabel("$\phi$")
    # plt.ylabel("$\psi$");
    # plt.show()


if __name__ == "__main__":
    run()