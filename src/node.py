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
        x0 (float): The minimum x-coordinate of the node's rectangle.
        x1 (float): The maximum x-coordinate of the node's rectangle.
        y0 (float): The minimum y-coordinate of the node's rectangle.
        y1 (float): The maximum y-coordinate of the node's rectangle.
        particle_indices (list): A list of indices of the particles contained in the node.
        children (dict): A dictionary containing references to the four child nodes of the current node (nw, ne, sw, se).
        com (numpy.ndarray): The center of mass of the node.
        total_mass (int): The total mass of the particles contained in the node.
    """
    def __init__(self, parent, rect: tuple, particle_data:ParticleData) -> None:
        self.parent = parent
        self.x0,self.x1,self.y0,self.y1 = rect
        self.particle_indices = []
        self.children = {"nw": None, "ne": None, "sw": None, "se": None}
        self.com = np.zeros(2, np.float32)
        self.total_mass = 0
        self.particle_data = particle_data

    def build_quadtree(self) -> None:
        """
        Builds the quadtree structure by initializing the root node and recursively subdiving it.

        Parameters:
            particle_data (ParticleData): An instance of ParticleData containing the particle information

        Returns:
            None
        """
        self.particle_indices = list(range(self.particle_data.num_particles))
        self._recursive_subdivision(self.particle_data)

    def update(self) -> None:
        """
        Update the quadtree by removing particles that have moved out of the node and reinserting them.

        Returns:
            None
        """
        reinsert_particles = []
        remove_particles = []

        for particle_index in self.particle_indices:
            particle = self.get_particle(particle_index)
            x,y = particle['position']
            if not self._contains(x,y) and self.parent == None:
                remove_particles.append(particle_index)
                self._recursively_remove_particle_from_nodes(particle_index)
            else:
                reinsert_particles.append(particle_index)
                self._recursively_remove_particle_from_nodes(particle_index)

        self._recursively_reinsert_particles(reinsert_particles)
        self.particle_data.remove(remove_particles)
        self._update_properties()
        for child_node in self.children.values():
            if child_node is not None:
                child_node.update()


    def _get_root_node(self):
        if self.parent == None:
            return self
        else:
            return self.parent._get_root_node()

    def _recursive_subdivision(self) -> None:
        """
        Recursively subdivides the current node into four child nodes and assigns the particles to the appropriate child nodes. Is meant to be called on the root node of a quadtree structure.

        Returns:
            None
        """
        if self._compute_density() > 1:
            x_mid = (self.x1 + self.x0) / 2
            y_mid = (self.y1 + self.y0) / 2

            self.children['nw'] = QuadtreeNode(self, (self.x0, x_mid, y_mid, self.y1))
            self.children['ne'] = QuadtreeNode(self, (x_mid, self.x1, y_mid, self.y1))
            self.children['sw'] = QuadtreeNode(self, (self.x0, x_mid, self.y0, y_mid))
            self.children['se'] = QuadtreeNode(self, (x_mid, self.x1, self.y0, y_mid))

            for particle_index in self.particle_indices:
                particle = self.particle_data.get_particle(particle_index)
                x, y = particle['position']
                child = self._contains(x,y)
                child.particle_indices.append(particle_index)
                child._recursive_subdivision()

                self.com += particle['position']
                self.total_mass += 1

        if self.total_mass > 0:
            self.com /= self.total_mass

    def _contains(self, x: int, y: int) -> 'QuadtreeNode':
        """
        Returns the child node that contains the given coordinates (x, y).

        Parameters:
            x : The x-coordinate of the point.
            y : The y-coordinate of the point.

        Returns:
            QuadtreeNode: The child node that contains the given coordinates (x, y).
        """
        root_node = self._get_root_node
        root_x0, root_x1, root_y0, root_y1 = root_node.x0, root_node.x1, root_node.y0, root_node.y1
        if x < root_x0 or x > root_x1 or y < root_y0 or y > root_y1:
            return None
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

    def _compute_density(self) -> float:
        """
        Calculate the density of a node in a 2D tree structure.

        Parameters
        ----------
        node : Node object
            The node for which the density needs to be computed. The node should have the following attributes:
                - x0 : float
                    The x-coordinate of the node's lower-left corner.
                - y0 : float
                    The y-coordinate of the node's lower-left corner.
                - x1 : float
                    The x-coordinate of the node's upper-right corner.
                - y1 : float
                    The y-coordinate of the node's upper-right corner.
                - total_mass : float
                    The total mass of particles contained within the node.

        Returns
        -------
        float
            The density of the node, computed as the total mass divided by the area of the node.
        """
        node_width = (self.x1 - self.x0)
        node_height = (self.y1 - self.y0)
        if (node_width) == 0 or (node_height) == 0:
            return 0.0
        else:
            return self.total_mass / (node_height * node_width)
        
    def _recursively_remove_particle_from_nodes(self, particle_index: int) -> None:
        if all(child is None for child in self.children.values()):
            return
        
        children = self.children.values()

        for child in children:
            if child is not None:
                particle_indices_copy = child.particle_indices.copy()
                for index in particle_indices_copy:
                    particle = self.particle_data.get_particle(index)
                    x,y = particle['position']
                    if not child._contains(x,y):
                        child.particle_indices.remove(index)

                self._recursively_remove_particle_from_nodes(particle_index)

    def _recursively_reinsert_particles(self, reinsert_particles: list[int]) -> None:
        """
        Reinserts the removed particles into the correct nodes based on their new positions.

        Parameters:
            reinsert_particles (List[int]): A list of particle indices that need to be reinserted.

        Returns:
            None
        """
        for particle_index in reinsert_particles:
            particle = self.particle_data.get_particle(particle_index)
            x, y = particle['position']
            node = self

            while True:
                if all(child is None for child in node.children.values()):
                    break

                child = node._contains(x, y)

                if child is None:
                    break

                node = child

            node.particle_indices.append(particle_index)
            
    
    def _update_properties(self):
        """
        Update the properties of the current node based on the new particle positions.

        Returns:
            None
        """
        self.com = np.zeros(2, dtype=float)
        self.total_mass = 0

        for particle_index in self.particle_indices:
            particle = self.particle_data.get_particle(particle_index)
            self.com += particle['position']
            self.total_mass += 1

        if self.total_mass > 0:
            self.com /= self.total_mass


        for child_node in self.children.values():
            if child_node is not None:
                child_node._update_properties(self.particle_data)
        