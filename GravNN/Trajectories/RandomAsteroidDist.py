import os
import GravNN
from GravNN.Trajectories.TrajectoryBase import TrajectoryBase
import pathlib
import numpy as np
import trimesh
from numba import njit, prange
from GravNN.Support.ProgressBar import ProgressBar

class RandomAsteroidDist(TrajectoryBase):
    def __init__(
        self, celestial_body, radius_bounds, points, model_file=None, **kwargs
    ):
        """A sample distribution which can sample from altitudes all the way down to the surface of the body.

        This is unlike the RandomDist class which samples randomly between the radius bounds without accounting
        for if the point exists within the body or not. As such this is generally most useful when generating
        distributions around irregularly shaped asteroids / bodies.

        Args:
            celestial_body (Celestial Body): planet about which samples are collected
            radius_bounds (list): upper and lower altitude bounds
            points (int): total number of samples
            model_file (str, optional): The path to the shape model. Defaults to None.
        """
        self.radius_bounds = radius_bounds

        if model_file is None:
            grav_file =  kwargs.get("grav_file", [None])[0] # asteroids grav_file is the shape model
            self.model_file = kwargs.get("shape_model", [grav_file])[0] # planets have shape model (sphere currently)
        else:
            self.model_file = model_file # if desire to overwrite default shape model
            
        filename, file_extension = os.path.splitext(self.model_file)

        self.shape_model = trimesh.load_mesh(self.model_file, file_type=file_extension[1:])
        self.points = points
        self.celestial_body = celestial_body

        # Reduce the search space to minimum altitude if it exists on the planet
        if self.radius_bounds[0] == 0.0:
            try: 
                self.radius_bounds[0] = celestial_body.min_radius
            except:
                pass

        super().__init__(**kwargs)

        pass

    def generate_full_file_directory(self):
        """Define the output directory based on number of points sampled,
        the radius/altitude limits, and the shape model used
        """
        try:
            model_name = os.path.basename(self.model_file).split('.')[0]
        except:
            model_name = str(self.model_file.split("Blender_")[1]).split(".")[0]
        self.trajectory_name = (
            os.path.splitext(os.path.basename(__file__))[0]
            + "/"
            + self.celestial_body.body_name
            + "N_"
            + str(self.points)
            + "_RadBounds"
            + str(self.radius_bounds)
            + "_shape_model"
            + model_name
        )
        self.file_directory += self.trajectory_name + "/"
        pass

    def generate(self):
        """Generate samples from uniform lat, lon, and radial distributions, but also check
        that those samples exist above the surface of the shape model. If not, resample the radial component

        Returns:
            np.array: cartesian positions of samples
        """
        X = []
        Y = []
        Z = []
        idx = 0
        X.extend(np.zeros((self.points,)).tolist())
        Y.extend(np.zeros((self.points,)).tolist())
        Z.extend(np.zeros((self.points,)).tolist())
        pbar = ProgressBar(self.points, enable=True)
        min_radius = np.max([self.radius_bounds[0], np.min(np.linalg.norm(self.shape_model.vertices,axis=1))*1000])
        while idx < self.points:
            angles = np.random.uniform(-1,1, size=(3,))
            angles /= np.linalg.norm(angles)
            s,t,u = angles
            r = np.random.uniform(min_radius, self.radius_bounds[1])
            X_inst = r * s
            Y_inst = r * t
            Z_inst = r * u

            distance = self.shape_model.nearest.signed_distance(
                np.array([[X_inst, Y_inst, Z_inst]]) / 1e3
            )
            # ensure that the point is outside of the body
            i = 0
            while distance > 0:
                # Note that this loop my get stuck if the radius bounds do not extend beyond the body
                # (i.e. the RA and Dec are fixed so if the upper bound does not extend beyond the shape
                # this criteria is never satisfied)
                r = np.random.uniform(min_radius, self.radius_bounds[1])
                X_inst = r * s
                Y_inst = r * t
                Z_inst = r * u
                distance = self.shape_model.nearest.signed_distance(
                    np.array([[X_inst, Y_inst, Z_inst]]) / 1e3
                )
                i += 1
                # Try to keep value at same RA and Dec, but if it can't find any then just select a new position entirely. 
                if i > 10:
                    angles = np.random.uniform(-1,1, size=(3,))
                    angles /= np.linalg.norm(angles)
                    s,t,u = angles
                    i = 0
            X[idx] = X_inst
            Y[idx] = Y_inst
            Z[idx] = Z_inst
            idx += 1
            pbar.update(idx)
        self.positions = np.transpose(np.array([X, Y, Z]))
        return np.transpose(np.array([X, Y, Z]))
