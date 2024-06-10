from sph_particle import SPHParticle
from typing import List
import numpy as np

def coffee_visualizer(mass: float) -> List:
    step = 0.75
    x_min, x_max = -5, 5
    y_min, y_max = -5, 5
    c_w = 1
    coffee_particles = []

    for i in range(11):
        for j in range(11):
            x = x_min + j * step
            y = y_min + i * step
            coffee_particles.append(SPHParticle(np.array([x, y]), [0.0, 0.0], mass, 'coffee', c_w))

    current_x = x_min + step * 11
    current_y = y_min + step
    for i in range(9):
        y = current_y + i * step
        coffee_particles.append(SPHParticle(np.array([current_x, y]), [0.0, 0.0], mass, 'coffee', c_w))

    current_x = x_min - step
    for i in range(9):
        y = current_y + i * step
        coffee_particles.append(SPHParticle(np.array([current_x, y]), [0.0, 0.0], mass, 'coffee', c_w))

    current_x = x_min + step
    current_y = y_min - step
    for i in range(9):
        x = current_x + i * step
        coffee_particles.append(SPHParticle(np.array([x, current_y]), [0.0, 0.0], mass, 'coffee', c_w))

    current_y = y_min + step * 11
    for i in range(9):
        x = current_x + i * step
        coffee_particles.append(SPHParticle(np.array([x, current_y]), [0.0, 0.0], mass, 'coffee', c_w))

    current_x = x_min + step * 12
    current_y = y_min + step * 3
    for i in range(5):
        y = current_y + i * step
        coffee_particles.append(SPHParticle(np.array([current_x, y]), [0.0, 0.0], mass, 'coffee', c_w))

    current_x = x_min - 2 * step
    for i in range(5):
        y = current_y + i * step
        coffee_particles.append(SPHParticle(np.array([current_x, y]), [0.0, 0.0], mass, 'coffee', c_w))

    current_x = x_min + 3 * step
    current_y = y_min - 2 * step
    for i in range(5):
        x = current_x + i * step
        coffee_particles.append(SPHParticle(np.array([x, current_y]), [0.0, 0.0], mass, 'coffee', c_w))

    current_y = y_min + step * 12
    for i in range(5):
        x = current_x + i * step
        coffee_particles.append(SPHParticle(np.array([x, current_y]), [0.0, 0.0], mass, 'coffee', c_w))

    return coffee_particles