import numpy as np

class ParticleData:
    """
    Class representing particle data.

    Parameters
    ----------
    num_particles : int
        The number of particles.
    dtype : float32
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
    mass : ndarray
        Array representing particle mass.
    """
    def __init__(self, num_particles: int, dtype=np.float32) -> None:
        self.num_particles = num_particles
        dtype = [
            ('position', dtype, 2),
            ('velocity', dtype, 2),
            ('force', dtype, 2),
            ('mass', dtype, 1)
        ]
        self.particles = np.zeros(num_particles, dtype)
        self.position = self.particles['position']
        self.velocity = self.particles['velocity']
        self.force = self.particles['force']
        self.mass = self.particles['mass']

    def initialize_particles(self, min_position:float, max_position:float, min_velocity:float, max_velocity:float) -> None:
        """
        Initialize particle positions and velocities with random values.

        Parameters
        ----------
        min_position : float
            The minimum value for particle positions.
        max_position : float
            The maximum value for particle positions.
        min_velocity : float
            The minimum value for particle velocities.
        max_velocity : float
            The maximum value for particle velocities.

            The bounds for these parameters is defined by the border height and width for animation, although it might make more sense to create a different distribution of particles then randomly across the whole screen.
        """
        self.position[:, 0] = np.random.uniform(min_position, max_position + 1, size = self.num_particles)
        self.position[:, 1] = np.random.uniform(min_position, max_position + 1, size = self.num_particles)
        self.velocity[:, 0] = np.random.uniform(min_velocity, max_velocity, size=self.num_particles)
        self.velocity[:, 1] = np.random.uniform(min_velocity, max_velocity, size=self.num_particles)

    def get_particle(self, index:int) -> np.ndarray:
        return self.particles[index]

    def integrate(self,dt) -> None:
        """
        Perform integration to update the particle positions and velocities.

        Parameters:
            dt (float): The time step for the integration.

        Returns:
            None
        """
        self.position += self.velocity * dt + 0.5 * self.force * dt**2
        
