import numpy as np
from particledata import ParticleData


class QuadtreeNode:
    """
    A class that represents a node in a quadtree.

    Parameters:
        parent (QuadtreeNode): The parent node of the current node.
        rect (tuple): A tuple representing the boundaries of the node's rectangle in the form (x0, x1, y0, y1).

    Attributes:
        parent (QuadtreeNode): The parent node of the current node.
        x0 (int): The minimum x-coordinate of the node's rectangle.
        x1 (int): The maximum x-coordinate of the node's rectangle.
        y0 (int): The minimum y-coordinate of the node's rectangle.
        y1 (int): The maximum y-coordinate of the node's rectangle.
        particle_indices (list): A list of indices of the particles contained in the node.
        children (dict): A dictionary containing references to the four child nodes of the current node (nw, ne, sw, se).
        com (numpy.ndarray): The center of mass of the node.
        total_mass (int): The total mass of the particles contained in the node.
    """
    def __init__(self, parent, rect: tuple) -> None:
        self.parent = parent
        self.x0,self.x1,self.y0,self.y1 = rect
        self.particle_indices = []
        self.children = {"nw": None, "ne": None, "sw": None, "se": None}
        self.com = np.zeros(2, int)

    def build_quadtree(self, particle_data:ParticleData) -> None:
        """
        Builds the quadtree structure by initializing the root node and recursively subdiving it.

        Parameters:
            particle_data (ParticleData): An instance of ParticleData containing the particle information

        Returns:
            None
        """
        self.particle_indices = list(range(particle_data.num_particles))
        self._subdivide(particle_data)

    def _subdivide(self, particle_data: ParticleData) -> None:
        """
        Recursively subdivides the current node into four child nodes and assigns the particles to the appropriate child nodes. Is meant to be called on the root node of a quadtree structure.

        Parameters:
            particle_data (ParticleData): An instance of ParticleData containing the particle information.

        Returns:
            None
        """
        if len(self.particle_indices) > 1:
            x_mid = (self.x1 + self.x0) / 2
            y_mid = (self.y1 + self.y0) / 2

            self.children['nw'] = QuadtreeNode(self, (self.x0, x_mid, y_mid, self.y1))
            self.children['ne'] = QuadtreeNode(self, (x_mid, self.x1, y_mid, self.y1))
            self.children['sw'] = QuadtreeNode(self, (self.x0, x_mid, self.y0, y_mid))
            self.children['se'] = QuadtreeNode(self, (x_mid, self.x1, self.y0, y_mid))

            for particle_index in self.particle_indices:
                particle = particle_data.get_particle(particle_index)
                x, y = particle['position']
                child = self._contains(x,y)
                child.particle_indices.append(particle_index)
                child._subdivide(particle_data)

                self.com += particle['position']

        if len(self.particle_indices) != 0:
            self.com /= len(self.particle_indices)

    def _contains(self, x: int, y: int) -> 'QuadtreeNode':
        """
        Returns the child node that contains the given coordinates (x, y).

        Parameters:
            x : The x-coordinate of the point.
            y : The y-coordinate of the point.

        Returns:
            QuadtreeNode: The child node that contains the given coordinates (x, y).
        """
        x_mid = (self.x0 + self.x1) / 2
        y_mid = (self.y0 + self.y1) / 2

        if x <= x_mid:
            if y >= y_mid:
                return self.children['nw']
            else:
                return self.children['sw']
        else:
            if y >= y_mid:
                return self.children['ne']
            else:
                return self.children['se']
