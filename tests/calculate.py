def calculate(self, particle_index: int, G: float, theta: float) -> np.ndarray:
    """
    Calculate the net force acting on a particle in the quadtree.

    Parameters:
        particle_index (int): The index of the particle for which to calculate the net force.
        G (float): The gravitational constant.
        theta (float): The threshold value for the Barnes-Hut approximation.

    Returns:
        numpy.ndarray: The net force acting on the particle.
    """
    particle = self.particle_data.get_particle(particle_index)
    position = particle['position']
    net_force = np.zeros(2)

    if self.particle is not None and self.particle_data.get_particle_index(self.particle) != particle_index:
        # Calculate the gravitational force between the target particle and the particle in the current node
        force = self._compute_gravitational_force(particle, self.particle, G)
        net_force += force

    if all(child is None for child in self.children.values()):
        # Leaf node (external node) containing only one particle
        return net_force

    # Compute the s/d ratio for the current node
    s_d_ratio = self._compute_theta(position)

    if s_d_ratio < theta:
        # Approximate the bodies in the node as a single body and calculate the gravitational force
        approx_mass = self.total_mass
        approx_position = self.com
        approx_particle = {'position': approx_position, 'mass': approx_mass}
        force = self._compute_gravitational_force(particle, approx_particle, G)
        net_force += force
    else:
        # Recursively traverse each child node
        for child_node in self.children.values():
            if child_node is not None:
                force = child_node.calculate(particle_index, G, theta)
                net_force += force

    return net_force
