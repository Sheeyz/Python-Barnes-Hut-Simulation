from node import QuadtreeNode
from particledata import ParticleData
import sys
import time
dt = 1/60
def initialize_quadtree_from_data(num_particles: int, x_min: int, x_max:int, y_min:int, y_max:int) -> QuadtreeNode:
    """
    Builds a quadtree structure from particle data.

    Parameters:
        num_particles (int): The number of particles.
        x_min (int): The minimum x-coordinate of the particle distribution.
        x_max (int): The maximum x-coordinate of the particle distribution.
        y_min (int): The minimum y-coordinate of the particle distribution.
        y_max (int): The maximum y-coordinate of the particle distribution.

    Returns:
        QuadtreeNode: The root node of the quadtree structure.
    """
    particle_data = ParticleData(num_particles)
    particle_data.initialize_particles(x_min, x_max, y_min, y_max)

    root_node = QuadtreeNode(parent=None, rect=(x_min, x_max, y_min, y_max), particle_data=particle_data)
    root_node.build_quadtree()

    return root_node

root_node = initialize_quadtree_from_data(200, 0, 800, 0, 800)

