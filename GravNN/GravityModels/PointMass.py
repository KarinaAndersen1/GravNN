import os

import numpy as np
from GravNN.GravityModels.GravityModelBase import GravityModelBase
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Support.transformations import cart2sph, invert_projection


def get_pm_data(trajectory, gravity_file, **kwargs):

    # Handle cases where the keyword wasn't properly wrapped as a list []
    override = bool(kwargs.get("override", [False])[0])

    point_mass_r0_gm = PointMass(kwargs['planet'][0], trajectory=trajectory)
    accelerations = point_mass_r0_gm.load(override=override).accelerations
    potentials = point_mass_r0_gm.potentials

    x = point_mass_r0_gm.positions  # position (N x 3)
    a = accelerations
    u = np.array(potentials).reshape((-1, 1))  # potential (N x 1)

    return x, a, u

class PointMass(GravityModelBase):
    def __init__(self, celestial_body, trajectory=None):
        """Gravity model that only produces accelerations and potentials
        as if there were only a point mass.

        Args:
            celestial_body (CelestialBody): body used to generate gravity measurements
            trajectory (TrajectoryBase, optional): trajectory for which gravity measurements must be produced. Defaults to None.
        """
        super().__init__()
        self.configure(trajectory)

        self.celestial_body = celestial_body
        self.mu = celestial_body.mu

    def generate_full_file_directory(self):
        self.file_directory += (
            os.path.splitext(os.path.basename(__file__))[0] + "_PointMass" + "/"
        )
        pass

    def compute_acceleration(self, positions=None):
        "Compute the acceleration for an existing trajectory or provided set of positions"
        if positions is None:
            positions = self.trajectory.positions

        positions = cart2sph(positions)
        self.accelerations = np.zeros(positions.shape)
        for i in range(len(self.accelerations)):
            self.accelerations[i] = self.compute_acceleration_value(positions[i])

        # accelerations are in spherical coordinates in the hill frame. Need to change to inertial frame
        self.accelerations = invert_projection(positions, self.accelerations)
        return self.accelerations

    def compute_potential(self, positions=None):
        "Compute the potential for an existing trajectory or provided set of positions"
        if positions is None:
            positions = self.trajectory.positions

        positions = cart2sph(positions)
        self.potentials = np.zeros(len(positions))
        for i in range(len(self.potentials)):
            self.potentials[i] = self.compute_potential_value(positions[i])

        return self.potentials

    def compute_acceleration_value(self, position):
        # remember that a = -dU/dx
        # U = -mu/r
        # dU/dx = mu/r^2
        # a = -dU/dx = -mu/r^2
        return np.array(
            [-self.mu / position[0] ** 2, 0, 0]
        )  # [a_r, theta, phi] -- theta and phi are needed to convert back to cartesian

    def compute_potential_value(self, position):
        return -self.mu / position[0]


def main():
    import time

    start = time.time()
    planet = Earth()
    point_mass = PointMass(planet)
    print(time.time() - start)

    position = (
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]]) * planet.radius
    )  # Must be in meters

    print(position)
    start = time.time()
    accelerations = point_mass.compute_acceleration(position)
    print(accelerations)
    print(np.linalg.norm(accelerations, axis=1))
    print(time.time() - start)

    from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics

    model = SphericalHarmonics(planet.sh_file, 1000)
    sh_results = model.compute_acceleration(position)

    print(sh_results)


if __name__ == "__main__":
    main()
