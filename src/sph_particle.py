import numpy as np
class SPHParticle:
    def __init__(self, position: np.ndarray, velocity: list[float], mass: float,
                 particle_type: str, c_w: float = 0.0):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.mass = mass
        self.type = particle_type
        self.c_w = c_w
        self.density = 0
        self.pressure = 0

    def update(self, dt: float, forces: np.ndarray) -> None:
        acceleration = forces / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt