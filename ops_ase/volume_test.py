from openpathsampling.ensembles.visit_all_states import *
# from __future__ import print_function
from typing import List
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.md import VelocityVerlet, MDLogger, Langevin
import ase.units as units

from ops_tutorial.ops_ase_test.engine import AseEngine

import openpathsampling as ops


def get_states():
    atoms = read(filename=Path(__file__).parent.parent / "AD_initial_frame.pdb")
    atoms.calc = atoms
    hi_T_integrator = Langevin(atoms=atoms, timestep=1 * units.fs, temperature_K=300, friction=0.01, logfile='./md.log')

    engine_options = {
        'n_steps_per_frame': 10,
        'n_frames_max': 100
    }
    hi_T_engine = AseEngine(
        hi_T_integrator,
        options=engine_options
    )
    current_snapshot = hi_T_engine.current_snapshot
    # define the CVs

    def f(atoms, indices):
        return atoms.get_dihedrals(indices)

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
    deg = 180.0 / np.pi
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


def test_default_state_progress_report():
    C_7eq, alpha_R = get_states()
    n_steps = 100
    found_vol = [C_7eq]
    all_vol  = [C_7eq, alpha_R]


    f = default_state_progress_report  # keep on one line

    # print(f(n_steps, found_vol, all_vol) == f"Ran 100 frames. Found states [{[element.name for element in found_vol]}]. "
    #                                         f"Looking for [{[element.name for element in all_vol if element not in found_vol]}].\n")
    print(f(n_steps, found_vol, all_vol))
test_default_state_progress_report()