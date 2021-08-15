from typing import List

from openpathsampling.engines.snapshot import BaseSnapshot
from openpathsampling.engines.features.base import attach_features
from openpathsampling.engines import features

from ase.md.md import MolecularDynamics

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

    def get_dihedrals(self, indices: List[List[float]]) -> float:
        """
        Calculate dihedral angles (in degrees) between the list of vectors a0->a1 and a2->a3
        indices: in the format[[a0, a1, a2, a3]]
        """
        return self.integrator.atoms.get_dihedrals(indices=indices, mic=False)