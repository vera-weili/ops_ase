from unittest import TestCase, mock

from openpathsampling.engines.snapshot import BaseSnapshot
from openpathsampling.engines.features.base import attach_features
from openpathsampling.engines import features

from ase.md.md import MolecularDynamics

from ops_ase.snapshot import AseSnapshot


class AseSanpshotTest(TestCase):
    def setUp(self) -> None:
        self.mock_integrator = mock.Mock()
        self.mock_integrator.atoms.get_dihedrals.return_value = 1

    def test_initialization(self):


@attach_features([
    features.engine,
    features.velocities,
    features.coordinates,
    features.box_vectors
])
class AseSnapshot(BaseSnapshot):
    """Fully functional snapshot"""
    def __init__(self, integrator: MolecularDynamics, velocities=None, coordinates=None, engine=None):
        self.integrator = integrator
        self.coordinates = coordinates
        self.velocities = velocities
        self.engine = engine

        self.__uuid__ = self.get_uuid()

        # super(MySnapshot, self).__init__() # don't inherit from BaseSnapshot because we don't want to use topology

    # @property
    # def topology(self):
    #     return self.engine.topology

    def get_dihedrals(self, indices):
        return self.integrator.atoms.get_dihedrals(indices=indices, mic=False)