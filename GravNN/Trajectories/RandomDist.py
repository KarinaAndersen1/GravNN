import os
from GravNN.Trajectories.TrajectoryBase import TrajectoryBase
import pathlib
import numpy as np
import trimesh

class RandomDist(TrajectoryBase):
    def __init__(self, celestial_body, radius_bounds, points, **kwargs):
        """A distribution that samples uniformly in a spherical volume.

        Args:
            celestial_body (Celestial Body): Planet about which samples should be taken
            radius_bounds (list): range of radii from which the sample can be drawn
            points (int): number of samples
        """
        self.radius_bounds = radius_bounds
        self.points = int(points)
        self.celestial_body = celestial_body

        uniform_volume = kwargs.get('uniform_volume', False)
        self.uniform_volume = uniform_volume[0] if isinstance(uniform_volume, list) else uniform_volume
        
        self.populate_shape_model(**kwargs)
        super().__init__(**kwargs)

        pass

    def populate_shape_model(self, **kwargs):
        try: 
            self.model_file = self.celestial_body.shape_model
        except:
            grav_file =  kwargs.get("grav_file", [None])[0] # asteroids grav_file is the shape model
            self.model_file = kwargs.get("shape_model", [grav_file]) # planets have shape model (sphere currently)  
            if isinstance(self.model_file, list):
                self.model_file = self.model_file[0]
        filename, file_extension = os.path.splitext(self.model_file)
        self.shape_model = trimesh.load_mesh(self.model_file, file_type=file_extension[1:])


    def generate_full_file_directory(self):
        directory_name = os.path.splitext(os.path.basename(__file__))[0]
        body = self.celestial_body.body_name
        try:
            model_name = os.path.basename(self.model_file).split('.')[0]
        except:
            model_name = str(self.model_file.split("Blender_")[1]).split(".")[0]
        self.trajectory_name = f"{directory_name}/{body}_{model_name}_N_{int(self.points)}_RadBounds{self.radius_bounds}_UVol_{self.uniform_volume}"
        self.file_directory += self.trajectory_name + "/"

    def sample_volume(self, points):
        X = []
        Y = []
        Z = []
        X.extend(np.zeros((points,)).tolist())
        Y.extend(np.zeros((points,)).tolist())
        Z.extend(np.zeros((points,)).tolist())

        theta = np.random.uniform(0, 2*np.pi, size=(points,))
        cosphi = np.random.uniform(-1,1, size=(points,))
        R_min = self.radius_bounds[0]
        R_max = self.radius_bounds[1]

        if self.uniform_volume:
            #https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
            u_min = (R_min / R_max)**3
            u_max = 1.0

            # want distribution to be uniform across volume the sphere
            u = np.random.uniform(u_min,u_max, size=(points,))

            # convert the uniform volume length into physical radius 
            r = R_max * u**(1.0/3.0)
        else:
            r = np.random.uniform(R_min,R_max, size=(points,))
        phi = np.arccos(cosphi)

        X = r * np.sin(phi) * np.cos(theta)
        Y = r * np.sin(phi) * np.sin(theta)
        Z = r * np.cos(phi)

        return np.transpose(np.array([X, Y, Z])) # [N x 3]

    def identify_interior_points(self, positions):
        mask = np.full((len(positions),), False)
        rayObject = trimesh.ray.ray_triangle.RayMeshIntersector(self.shape_model)
        mask = rayObject.contains_points(positions / 1E3) 
        return mask

    def recursively_remove_interior_points(self, positions):
        mask = self.identify_interior_points(positions)
        interior_points = np.sum(mask)
        print(f"Remaining Points: {interior_points}")
        if interior_points > 0:
            new_positions = self.sample_volume(interior_points)
            positions[mask] = self.recursively_remove_interior_points(new_positions)
        return positions


    def generate(self):
        """Randomly sample from uniform latitude, longitude, and radial distributions

        Returns:
            np.array: cartesian positions of the samples
        """
        positions = self.sample_volume(self.points)
        positions = self.recursively_remove_interior_points(positions)
        self.positions = positions
        return positions.copy()

if __name__ == "__main__":
    from GravNN.CelestialBodies.Asteroids import Eros
    from GravNN.CelestialBodies.Planets import Earth
    # traj = RandomDist(Eros(), [0, Eros().radius], 10000, shape_model=Eros().obj_8k, override=[False])
    traj = RandomDist(Earth(), [Earth().radius, Earth().radius*2], 10000)
