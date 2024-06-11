from typing import List
from sph_particle import SPHParticle
import numpy as np


class CoffeeGrain:
    def __init__(self, particles: List[SPHParticle]):
        self.particles = particles
    def find_neighbours(self, particle: SPHParticle, radius: float = 1.0) -> List[SPHParticle]:
        neighbors = []
        for other in self.particles:
            if np.linalg.norm(particle.position - other.position) < radius:
                neighbors.append(other)
        return neighbors