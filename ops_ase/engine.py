from typing import Dict, Optional
from copy import deepcopy

from openpathsampling.engines import DynamicsEngine as OpsDynamicsEngine
from openpathsampling.engines import SnapshotDescriptor
from ase.md.md import MolecularDynamics as AseMolecularDynamics

from ops_ase.snapshot import AseSnapshot


class AseEngine(OpsDynamicsEngine):

    base_snapshot_type = AseSnapshot
    _default_options = {
        'n_steps_per_frame': 10,
        'n_frames_max': 100,
    }

    def __init__(self, topology, integrator: AseMolecularDynamics, options: Optional[Dict] = None):
        """
            options : dict of { str : value }, a dictionary
        """
        self.integrator = integrator
        self.topology = topology

        dimensions = {
            'n_atoms': topology.n_atoms,
            'n_spatial': topology.n_spatial
        }

        descriptor = SnapshotDescriptor.construct(
            snapshot_class=AseSnapshot,
            snapshot_dimensions=dimensions
        )

        super(AseEngine, self).__init__(options=options, descriptor=descriptor)


        self.positions = None
        self.velocities = None
        # self.topology = None
        self._current_snapshot = None
        self._update_current_snapshot()

    @property
    def current_snapshot(self) -> AseSnapshot:
        """
        current_snapshot velocities and coordinates are obtained from ase Atoms object
        Returns
        -------
        """
        return self._current_snapshot

    @current_snapshot.setter
    def current_snapshot(self, snapshot: AseSnapshot) -> None:
        self.check_snapshot_type(snapshot)
        self.positions = snapshot.coordinates
        self.velocities = snapshot.velocities
        self._current_snapshot = self._update_current_snapshot()

    def _update_current_snapshot(self) -> None:
        """
        Update self._current_snapshot using self.integrator
        """
        atoms_object = deepcopy(self.integrator.atoms)
        self._current_snapshot = AseSnapshot(atoms=atoms_object)
        # self._current_snapshot = AseSnapshot(
        #     integrator=self.integrator,
        #     velocities=atoms_object.get_velocities,
        #     coordinates=atoms_object.get_positions,
        #     engine=self)

    @property
    def n_steps_per_frame(self) -> int:
        return self.options['n_steps_per_frame']

    @n_steps_per_frame.setter
    def n_steps_per_frame(self, value: int) -> None:
        self.options['n_steps_per_frame'] = value

    @property
    def snapshot_timestep(self):
        return self.n_steps_per_frame * self.integrator.dt

    def generate_next_frame(self) -> AseSnapshot:
        """

        Returns: a series of current_snapshot to form a trajectory
        -------

        """
        # self.integrator.run(self.n_steps_per_frame)
        # for _ in range(self.n_steps_per_frame):
        self.integrator.step()
        self._update_current_snapshot()
        return self.current_snapshot

    # def _velocities_update(self, delta_t):  # ToDo: decide how to update velocites
    #     pass
    #
    # def _positions_update(self, snapshot: SnapShot, delta_t: float):  # ToDo: decide how to update velocites
    #     """
    #     Parameters
    #     ----------
    #     snapshot
    #     delta_t: in ps
    #         time step
    #
    #     Returns
    #     -------
    #
    #     """
    #     snapshot.positions += snapshot.velocities * delta_t
    #
    # def step(self, snapshot):
    #     """
    #     Parameters
    #     ----------
    #     snapshot: an MD step. update in-place. snapshot contains its state, including velocities and masses
    #
    #     Returns: next snapshot momentum and position
    #     -------
    #
    #     """
    #     self._positions_update(snapshot, 0.5*self.delta_t)  # ToDo: decide how to set/use self.dt
    #     self._positions_update(snapshot, 0.5 * self.integrator.timestep)  # ToDo: decide how to set/use self.dt, or use itegrator's timestep?
    #     self._velocities_udpate(snapshot, self.delta_t)

