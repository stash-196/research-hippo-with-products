class L63:
    """
    Implements the Lorenz 63 system with methods to step through
    and store the trajectory history.
    """

    def __init__(self, sigma, rho, beta, init, dt):
        """
        sigma, rho, beta : Lorenz 63 parameters
        init : initial [x, y, z]
        dt : time step
        """
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.x, self.y, self.z = init
        self.dt = dt
        self.hist = [init]

    def step(self):
        """Euler step of the Lorenz 63 system."""
        self.x += self.sigma * (self.y - self.x) * self.dt
        self.y += (self.x * (self.rho - self.z)) * self.dt
        self.z += (self.x * self.y - self.beta * self.z) * self.dt
        self.hist.append([self.x, self.y, self.z])

    def integrate(self, n_steps):
        """Repeatedly calls step()."""
        for _ in range(n_steps):
            self.step()
