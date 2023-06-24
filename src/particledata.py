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
    """
    def __init__(self, num_particles: int, dtype=np.float32) -> None:
        self.num_particles = num_particles

        dtype = [
            ('position', dtype, 2),
            ('velocity', dtype, 2),
            ('force', dtype, 2),
            ('mass', dtype, 1),
            ('previous_position', dtype, 2)
        ]

        self.particles = np.zeros(num_particles, dtype)
        self.position = self.particles['position']
        self.velocity = self.particles['velocity']
        self.force = self.particles['force']
        self.mass = self.particles['mass']
        self.previous_position = self.particles['previous_position']

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
        """
        Retrieve the particle data at the specified index.

        Parameters:
            index (int): Index of the particle.

        Returns:
            np.ndarray: Particle data at the given index.
        """
        return self.particles[index]
    
    def integrate(self,dt) -> None:
        """
        Perform integration to update the particle positions and velocities given the force acting on them.

        Parameters:
            dt (float): The time step for the integration.

        Returns:
            None
        """
        for particle_index in range(self.num_particles):
            particle = self.get_particle(particle_index)
            mass = particle['mass']
            force = particle['force']
            position = particle['position']
            previous_position = particle['previous_position']
            if mass != 0:
                acceleration = force / mass
            else:
                acceleration = np.zeros(2)
            position_new = 2 * position - previous_position + acceleration * dt**2
            particle['previous_position'] = position
            particle['position'] = position_new
            velocity_new = (position_new - previous_position) / (2 * dt)
            particle['velocity'] = velocity_new