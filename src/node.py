import numpy as np
from particledata import ParticleData

dt = 1/60


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
        particle_indices (dict): A dictionary of indices of the particles mapped to the node they are contained in.
        children (dict): A dictionary containing references to the four child nodes of the current node (nw, ne, sw, se).
        com (numpy.ndarray): The center of mass of the node.
        total_mass (float): The total mass of the particles contained in the node.
    """
    particle_indices = {}
    def __init__(self, parent, rect: tuple, particle_data:ParticleData) -> None:
        self.parent = parent
        self.x0,self.x1,self.y0,self.y1 = rect
        self.children = {"nw": None, "ne": None, "sw": None, "se": None}
        self.particle_data = particle_data
        self.theta = 0.5
        self.num_particles = particle_data.num_particles

    def build_quadtree(self) -> None:
        """
        Builds the quadtree structure by initializing the root node and recursively subdiving it.

        Returns:
            None
        """
        QuadtreeNode.particle_indices = {i: self for i in range(self.particle_data.num_particles)}
        self._recursive_subdivision()

    def update(self) -> None:
        """
        Update the quadtree by removing particles that have moved out of the node and reinserting them.

        Returns:
            None
        """
        self.particle_data = self.calculate_forces()
        self.particle_data.integrate(dt)

        for child_node in self.children.values():
            if child_node is not None:
                child_node.update()


    def _get_root_node(self):
        if self.parent is None:
            return self
        else:
            return self.parent._get_root_node()
        

    def _is_leaf(self):
        if all(child is None for child in self.children):
            return 1
        return 0
        

    def _recursive_subdivision(self) -> None:
        """
        Recursively subdivides the current node into four child nodes and assigns the particles to the appropriate child nodes. Is meant to be called on the root node of a quadtree structure.

        Returns:
            None
        """
        if self.num_particles > 1:
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
                self.particle_indices[particle_index] = child
                self.num_particles += 1

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
        root_node = self._get_root_node()
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

    def _compute_theta(self, particle_index) -> float:
        """
        Calculate theta which is the ratio node_size / distance from particle to node center of mass.

        Returns:
            float: The value theta which determines the accuracy of the computation.
        """
        node_width = self.x1 - self.x0
        node_height = self.y1 - self.y0
        node_size = max(node_width, node_height)
        position_of_particle = self.particle_data.get_particle(particle_index)['position']
        dx = self.com[0] - position_of_particle[0]
        dy = self.com[1] - position_of_particle[1]
        distance = np.sqrt(dx**2 + dy**2)
        theta = node_size / distance
        return theta    

    def calculate_forces(self):
         """
    Calculate the forces acting on particles using the Barnes-Hut algorithm and the quadtree structure.

    Returns:
        ParticleData: The updated particle data with calculated forces.
        """
         particle_data = self.particle_data

         for particle_index in range(particle_data.num_particles):
             particle = particle_data.get_particle(particle_index)
             x, y = particle['position']
             force = np.zeros(2, dtype=np.float32)

             self._calculate_force(particle_index, x, y, force)
         
             particle['force'] = force
         
         return particle_data
    
    def _calculate_force(self, particle_index: int, x: float, y:float, force:np.ndarray):
        """
        Calculate the net force acting on a particle recursively by traversing the quadtree.

        Parameters:
            particle_index (int): The index of the particle for which to calculate the force.
            x (float): The x-coordinate of the particle.
            y (float): The y-coordinate of the particle.
            force (numpy.ndarray): The array to store the calculated net force.

        Returns:
            None
        """
        if self is None:
            return
        
        if self.particle_indices and len(self.particle_indices) == 1 and self.particle_indices[0] == particle_index:
            return
        
        if self._compute_theta(particle_index) < 0.5:
            particle = self.particle_data.get_particle(particle_index)
            dx = self.com[0] - x
            dy = self.com[1] - y
            r_squared = dx ** 2 + dy ** 2
            force_magnitude = particle['mass'] / r_squared
            force[0] += force_magnitude * dx
            force[1] += force_magnitude * dy

        else:
            for child_node in self.children.values():
                if child_node is not None:
                    child_node._calculate_force(particle_index, x, y, force)


    def resolve_movement(self):
        for particle_index, node in self.particle_indices.items():
            particle = self.particle_data.get_particle(particle_index)
            x,y = particle['position']
            while not node._contains(x,y) and node.parent is not None:   
                node = node.parent
            while not node._is_leaf():
                child = node._contains(x,y)
                if child is None:
                    break
                node = child

            self.particle_indices[particle_index] = node

        duplicates = self.check_duplicate_nodes(self.particle_data)
        for node in duplicates:
            node._recursivde_subdivide()

        
    def check_duplicate_nodes(self, particle_indices):
        # Create a dictionary to track the count of each node
        node_counts = {}
        
        # Iterate over the particle_indices dictionary
        for node in particle_indices.values():
            # Increment the count for the node
            node_counts[node] = node_counts.get(node, 0) + 1
        
        # Check if any node has more than one particle
        duplicate_nodes = [node for node, count in node_counts.items() if count > 1]
        
        return duplicate_nodes