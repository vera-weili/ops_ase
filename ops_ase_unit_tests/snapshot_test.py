from unittest import TestCase, mock

from ops_ase.snapshot import AseSnapshot


class AseSanpshotTest(TestCase):
    def setUp(self) -> None:
        self.mock_integrator = mock.Mock()
        self.mock_integrator.atoms.get_dihedrals.return_value = 1

    def test_initialization(self):
        snapshot = AseSnapshot(integrator=self.mock_integrator)
        self.assertIsNone(snapshot.velocities)
        self.assertIsNone(snapshot.coordinates)
        self.assertIsNone(snapshot.engine)

    def test_get_dihedrals(self):
        snapshot = AseSnapshot(integrator=self.mock_integrator)
        dihedrals = snapshot.get_dihedrals(indices=[[0, 1, 2, 3]])
        self.assertEqual(dihedrals, 1)
