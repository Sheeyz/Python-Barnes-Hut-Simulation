import node
from particledata import ParticleData
import numpy as np
import sys
import time

def initialize_quadtree_from_data(num_particles: int, size_min: int, size_max:int, vel_min:int, vel_max:int) -> node.QuadtreeNode:
    """
    Builds a quadtree structure from particle data.

    Parameters:
        num_particles (int): The number of particles.
        size_min (int): The minimum coordinate of the particle distribution.
        size_max (int): The maximum coordinate of the particle distribution.
        vel_min (int): The minimum velocity of the particle distribution.
        vel_max (int): The maximum velocity of the particle distribution.

    Returns:
        QuadtreeNode: The root node of the quadtree structure.
    """
    particle_data = ParticleData(num_particles)
    particle_data.initialize_particles(size_min, size_max, vel_min, vel_max)

    root_node = node.QuadtreeNode(parent=None, rect=(size_min, size_max, size_min, size_max), particle_data=particle_data)
    root_node.build_quadtree()

    return root_node

start = time.time()
root_node = initialize_quadtree_from_data(50000, 0, 10000, 0, 10)

for index in root_node.particle_indices:
    root_node.calculate(index, 6.6743e-11)
end = time.time()


print(end-start)
