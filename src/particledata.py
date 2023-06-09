import numpy as np

class ParticleData:
    """
    Class representing particle data.

    Parameters
    ----------
    num_particles : int
        The number of particles.
    dtype : int16
        The datatype for the values in the position, velocity, and force arrays.

    Attributes
    ----------
    num_particles : int
        The number of particles.
    particles : ndarray
        Array containing particle data.
    position : ndarray
        Array representing particle positions.
    velocity : ndarray
        Array representing particle velocities.
    force : ndarray
        Array representing particle forces.
    """
    def __init__(self, num_particles: int, dtype=np.int16) -> None:
        """
        Initialize ParticleData with the given number of particles and datatype.

        Parameters
        ----------
        num_particles : int
            The number of particles.
        dtype : numpy dtype, optional
            The datatype for the values in the position, velocity, and force arrays.
            Default is np.int16.
        """
        self.num_particles = num_particles

        dtype = [
            ('position', dtype, 2),
            ('velocity', dtype, 2),
            ('force', dtype, 2)
        ]

        self.particles = np.zeros(num_particles, dtype)
        self.position = self.particles['position']
        self.velocity = self.particles['velocity']
        self.force = self.particles['force']

    def initialize_particles(self, min_position:int, max_position: int, min_velocity:int, max_velocity:int) -> None:
        """
        Initialize particle positions and velocities with random values.

        Parameters
        ----------
        min_position : int
            The minimum value for particle positions.
        max_position : int
            The maximum value for particle positions.
        min_velocity : int
            The minimum value for particle velocities.
        max_velocity : int
            The maximum value for particle velocities.
        """
        self.position[:, 0] = np.random.randint(min_position, max_position + 1, size = self.num_particles)
        self.position[:, 1] = np.random.randint(min_position, max_position + 1, size = self.num_particles)
        self.velocity[:, 0] = np.random.randint(min_velocity, max_velocity, size=self.num_particles)
        self.velocity[:, 1] = np.random.randint(min_velocity, max_velocity, size=self.num_particles)

    def get_particle(self, index):
        return self.particles[index]