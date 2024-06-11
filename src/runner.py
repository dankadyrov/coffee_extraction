import numpy as np
from sph_particle import SPHParticle
from simulation import Simulation
from initial_conditions import coffee_visualizer
from kernels import kernel, kernel_derivative


def run():
    total_time = 5
    pressure_coefficient = 20.0
    viscosity_coefficient = 20.0
    coffee_water_viscosity = 75.0
    gravity = 1000.0
    rest_density = 1.0

    x_min, x_max = -7.5, 5
    y_min, y_max = 5, 15

    step = 0.5
    num_points_x = int((x_max - x_min) / step) + 1
    num_points_y = int((y_max - y_min) / step) + 1
    water_particles = []
    for i in range(num_points_y):
        for j in range(num_points_x):
            x = x_min + j * step
            y = y_min + i * step
            water_particles.append(SPHParticle(np.array([x, y]), [0.0, 1.0], 1.0, 'water'))

    print(len(water_particles))

    particles = coffee_visualizer(1.5, 5, 3)
    particles = list(np.append(particles, water_particles))

    sim = Simulation(particles, timestep=0.02, total_time=total_time, Dw=0.00001, kernel=kernel,
                     kernel_derivative=kernel_derivative, pressure_coefficient=pressure_coefficient,
                     viscosity_coefficient=viscosity_coefficient, coffee_water_viscosity=coffee_water_viscosity,
                     rest_density=rest_density, gravity=gravity)

    sim.run()
    sim.show_mean_concentration_per_time()