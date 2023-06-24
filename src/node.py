import numpy as np
from particledata import ParticleData


class QuadtreeNode:
    """
    A class that represents a node in a quadtree.

    Parameters:
        parent (QuadtreeNode): The parent node of the current node.
        rect (tuple): A tuple representing the boundaries of the node's rectangle in the form (x0, x1, y0, y1).
        particle_data (ParticleData): The class containing the array which houses the data for the particles.

    Attributes:
        parent (QuadtreeNode): The parent node of the current node.
        x0 (float): The minimum x-coordinate of the node's rectangle.
        x1 (float): The maximum x-coordinate of the node's rectangle.
        y0 (float): The minimum y-coordinate of the node's rectangle.
        y1 (float): The maximum y-coordinate of the node's rectangle.
        children (dict): A dictionary containing references to the four child nodes of the current node (nw, ne, sw, se).
        particle_data (ParticleData): The particle data that is contained within the tree.
        particle_indices (list): A list of indices of particles contained within a node.
        com (numpy.ndarray): The center of mass of the node.
        total_mass (float): The total mass of the particles contained in the node.
        size (float): The largest dimension of the box contained within the coordinates defined above.
    """
    def __init__(self, parent: 'QuadtreeNode', rect: tuple, particle_data: ParticleData) -> None:
        self.parent = parent
        self.x0,self.x1,self.y0,self.y1 = rect
        self.children = {"nw": None, "ne": None, "sw": None, "se": None}
        self.particle_data = particle_data
        self.particle_indices = []
        self.com = np.zeros(2)
        self.total_mass = 0
        self.size = max(self.x1-self.x0, self.y1-self.y0)

    def build_quadtree(self) -> None:
        """
        Builds the quadtree structure by initializing the root node and recursively subdiving it.

        Returns:
            None
        """
        self.particle_indices = [i for i in range(self.particle_data.num_particles)]
        self._recursive_subdivision()

    def _recursive_subdivision(self) -> None:
        """
        Recursively subdivides the current node into four child nodes and assigns the particles to the appropriate child nodes. Is meant to be called on the root node of a quadtree structure.

        Returns:
            None
        """
        if len(self.particle_indices) > 10:
            x_mid = (self.x1 + self.x0) / 2
            y_mid = (self.y1 + self.y0) / 2

            self.children['nw'] = QuadtreeNode(self, (self.x0, x_mid, y_mid, self.y1), self.particle_data)
            self.children['ne'] = QuadtreeNode(self, (x_mid, self.x1, y_mid, self.y1), self.particle_data)
            self.children['sw'] = QuadtreeNode(self, (self.x0, x_mid, self.y0, y_mid), self.particle_data)
            self.children['se'] = QuadtreeNode(self, (x_mid, self.x1, self.y0, y_mid), self.particle_data)


            for particle_index in self.particle_indices:
                particle = self.particle_data.get_particle(particle_index)
                x, y = particle['position']
                child = self._contains(x,y)
                child.particle_indices.append(particle_index)
                child.com += particle['position'] * particle['mass']
                child.total_mass += particle['mass']

            if child.total_mass != 0:
                child.com / child.total_mass

            for child in self.children.values():
                child._recursive_subdivision()


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

    def compute_theta(self, index, theta_max):
        """
        Compute theta value based on the given index and maximum theta.

        Parameters:
            index (int): Index of the particle.
            theta_max (float): Maximum theta value.

        Returns:
            bool: True if theta is less than theta_max, False otherwise.
        """
        particle = self.particle_data.get_particle(index)
        distance = np.sqrt(np.sum((particle['position']- self.com)**2))
        theta = self.size/distance
        return theta < theta_max
            
    def calculate(self, index, gravity):
        """
        Calculate the force exerted on a particle based on its index and gravity.

        Parameters:
            index (int): Index of the particle.
            gravity (float): Strength of gravity.

        Returns:
            None
        """
        particle = self.particle_data.get_particle(index)
        x,y = particle['position']
        if self._should_calculate_force(index) and not self._is_far_away(x, y):
            dx = particle['position'][0] - self.com[0]
            dy = particle['position'][1] - self.com[1]
            distance_squared = dx**2 + dy**2
            force = (gravity*self.total_mass * particle['mass'] / distance_squared)
            particle['force'][0] += force * dx / np.sqrt(distance_squared)
            particle['force'][1] += force * dy / np.sqrt(distance_squared)
        else:
            for child in self.children.values():
                if child is not None:
                    if child._is_far_away(x,y):
                        continue
                    child.calculate(index, gravity)
    
    def _should_calculate_force(self, index):
        """
        Check if the force should be calculated for a given particle index.

        Parameters:
            index (int): Index of the particle.

        Returns:
            bool: True if the force should be calculated, False otherwise.
        """
        return len(self.particle_indices) == 10 and index != self.particle_indices[0] or self.compute_theta(index, 1)

    def _is_far_away(self, x, y):
        """
        Check if the given coordinates are far away from the center of mass.

        Parameters:
            x (float): x-coordinate of the position.
            y (float): y-coordinate of the position.

        Returns:
            bool: True if the coordinates are far away, False otherwise.
        """
        dx = x - self.com[0]
        dy = y - self.com[1]
        distance = np.sqrt(dx ** 2 + dy ** 2)
        return distance > self.size
    
    