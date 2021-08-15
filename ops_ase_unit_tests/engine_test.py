from openpathsampling.engines import DynamicsEngine, features, SnapshotDescriptor
from ase.md.md import MolecularDynamics

from ops_ase.snapshot import AseSnapshot as SnapShot


class AseEngine(DynamicsEngine):

    base_snapshot_type = SnapShot
    _default_options = {
        'n_steps_per_frame': 10,
        'n_frames_max': 100,
    }

    def __init__(self, integrator: MolecularDynamics, options=None, topology=None):
        self.integrator = integrator
        self.topology = topology

        # descriptor = SnapshotDescriptor.construct(
        #     snapshot_class=SnapShot,
        #     snapshot_dimensions={'n_atoms': 1, 'n_spatial': 2}
        # )
        super(AseEngine, self).__init__(options=options)

        self.positions = None
        self.velocities = None
        # self.topology = None
        self._current_snapshot = self._update_current_snapshot()

    @property
    def current_snapshot(self) -> SnapShot:
        """
        current_snapshot velocities and coordinates are obtained from ase Atoms object
        Returns
        -------
        """
        return self._current_snapshot

    @current_snapshot.setter
    def current_snapshot(self, snapshot: SnapShot) -> None:
        self.check_snapshot_type(snapshot)
        self.positions = snapshot.coordinates
        self.velocities = snapshot.velocities
        self._current_snapshot = self._update_current_snapshot()

    def _update_current_snapshot(self) -> SnapShot:
        atoms_object = self.integrator.atoms
        return SnapShot(
            integrator=self.integrator,
            velocities=atoms_object.get_velocities,
            coordinates=atoms_object.get_positions,
            engine=self)

    @property
    def n_steps_per_frame(self) -> int:
        return self.options['n_steps_per_frame']

    @n_steps_per_frame.setter
    def n_steps_per_frame(self, value: int) -> None:
        self.options['n_steps_per_frame'] = value

    @property
    def snapshot_timestep(self):
        return self.n_steps_per_frame * self.integrator.dt

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

    def generate_next_frame(self) -> SnapShot:
        """

        Returns: a series of current_snapshot944- to form a trajectory
        -------

        """
        # self.integrator.run(self.n_steps_per_frame)
        self.integrator.step()
        self.current_snapshot = self.integrator.atoms
        return self.current_snapshot
