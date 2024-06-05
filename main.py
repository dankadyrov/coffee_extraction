import numpy as np
from matplotlib import pyplot as plt
from typing import Callable, List


class SPHParticle:
    def __init__(self, position: np.ndarray[float, float], velocity: list[float, float], mass: float,
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

    def update_position(self, dt: float) -> None:
        self.position += self.velocity * dt

    def apply_force(self, force: np.ndarray, dt: float) -> None:
        acceleration = force / self.mass
        self.velocity += acceleration * dt

    def update_c_w(self, Dw: float, dt: float, neighbors: List['SPHParticle']):
        for neighbor in neighbors:
            if neighbor.type == 'water':
                distance = np.linalg.norm(self.position - neighbor.position)
                self.c_w += Dw * (neighbor.c_w - self.c_w) / (distance ** 2) * dt


class CoffeeGrain:
    def __init__(self, particles: List[SPHParticle]):
        self.particles = particles

    def update_particles(self, dt: float) -> None:
        for particle in self.particles:
            particle.update_position(dt)

    def apply_forces(self, forces: List[np.ndarray], dt: float) -> None:
        for particle, force in zip(self.particles, forces):
            particle.apply_force(force, dt)

    def update_swelling(self, Dw: float, dt: float) -> None:
        for particle in self.particles:
            neighbors = self.find_neighbours(particle)
            particle.update_c_w(Dw, dt, neighbors)
            swelling_factor = (1 - particle.c_w) ** (-1 / 3)
            particle.position *= swelling_factor

    def find_neighbours(self, particle: SPHParticle, radius: float = 1.0) -> List[SPHParticle]:
        neighbors = []
        for other in self.particles:
            if np.linalg.norm(particle.position - other.position) < radius:
                neighbors.append(other)
        return neighbors


class Simulation:
    def __init__(self, grain: CoffeeGrain, water_particles: List[SPHParticle], timestep: float, total_time: float,
                 Dw: float, kernel: Callable, kernel_derivative: Callable, pressure_coefficient: float,
                 viscosity_coefficient: float, rest_density: float, gravity: float):
        self.grain = grain
        self.water_particles = water_particles
        self.timestep = timestep
        self.total_time = total_time
        self.current_time = 0
        self.Dw = Dw
        self.kernel = kernel
        self.kernel_derivative = kernel_derivative
        self.pressure_coefficient = pressure_coefficient
        self.viscosity_coefficient = viscosity_coefficient
        self.rest_density = rest_density
        self.gravity = gravity

    def run(self) -> None:
        while self.current_time < self.total_time:
            self.update()
            if int(round(self.current_time, 2) * 250) % 100 == 0:
                self.visualize(self.current_time)
            self.current_time += self.timestep

    def update(self) -> None:
        self.compute_densities()
        self.compute_pressure()
        forces = self.compute_forces()
        self.apply_forces_and_update(forces)
        self.grain.update_swelling(self.Dw, self.timestep)

    def apply_forces_and_update(self, forces: List[np.ndarray]) -> None:
        for particle, force in zip(self.water_particles, forces):
            particle.update(self.timestep, force)
        self.grain.apply_forces(forces, self.timestep)
        self.grain.update_particles(self.timestep)

    def compute_densities(self) -> None:
        for particle in self.grain.particles:
            density = 0
            for neighbor in self.grain.find_neighbours(particle):
                distance = np.linalg.norm(particle.position - neighbor.position)
                density += particle.mass * self.kernel(distance)
            particle.density = density

    def compute_pressure(self) -> None:
        for particle in self.grain.particles:
            particle.pressure = self.pressure_coefficient * (particle.density - self.rest_density)

    def compute_forces(self) -> List[np.ndarray]:
        forces = []
        for particle in self.grain.particles:
            force = np.array([0.0, 0.0])
            for neighbor in self.grain.find_neighbours(particle):
                distance = np.linalg.norm(particle.position - neighbor.position)
                if distance > 0:
                    pressure_force = -1.0 * self.pressure_coefficient * (particle.pressure + neighbor.pressure) / (
                            2 * neighbor.density) * self.kernel_derivative(distance)
                    viscosity_force = self.viscosity_coefficient * (
                            neighbor.velocity - particle.velocity) / neighbor.density * self.kernel(distance)
                    force += pressure_force + viscosity_force
            force[1] -= self.gravity * self.timestep
            forces.append(force)
        return forces

    def visualize(self, current_time: float) -> None:
        x_coffee = [p.position[0] for p in self.grain.particles if p.type == 'coffee']
        y_coffee = [p.position[1] for p in self.grain.particles if p.type == 'coffee']
        x_water = [p.position[0] for p in self.water_particles if p.type == 'water']
        y_water = [p.position[1] for p in self.water_particles if p.type == 'water']
        plt.figure(figsize=(12, 12))
        plt.scatter(x_coffee, y_coffee, color='brown', label='Coffee', alpha=0.5, s=10)
        plt.scatter(x_water, y_water, color='blue', label='Water', alpha=0.5, s=10)
        plt.title(f"Current time: {current_time:.2f}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.xlim(-50, 50)
        plt.ylim(-40, 40)
        plt.show(block=False)
        plt.pause(0.01)
        plt.close()


def main():
    h = 0.1
    pressure_coefficient = 1000.0
    viscosity_coefficient = 10.0
    rest_density = 1000.0

    def kernel(distance) -> float:
        q = distance / h
        sigma = 10 / (7 * np.pi * h ** 2)
        if q <= 1:
            return sigma * (1 - 1.5 * q ** 2 + 0.75 * q ** 3)
        elif q <= 2:
            return sigma * 0.25 * (2 - q) ** 3
        else:
            return 0

    def kernel_derivative(distance) -> float:
        q = distance / h
        sigma = 10 / (7 * np.pi * h ** 2)
        if q <= 1:
            return sigma * (-3 * q + 2.25 * q ** 2) / h
        elif q <= 2:
            return sigma * -0.75 * (2 - q) ** 2 / h
        else:
            return 0

    num_coffee_particles = 400
    num_water_particles = 300
    coffee_radius = 10.0
    gravity = 1000
    water_particles = [SPHParticle(np.random.random(2) * 20, [0.0, 0.0], 1.0, 'water') for _
                       in range(num_water_particles)]

    coffee_particles = []
    for _ in range(num_coffee_particles):
        r = coffee_radius * np.sqrt(np.random.uniform(0, 1))
        theta = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta) - coffee_radius - 5
        coffee_particles.append(SPHParticle(np.array([x, y]), [0.0, 0.0], 10.0, 'coffee'))

    grain = CoffeeGrain(coffee_particles)

    sim = Simulation(grain, water_particles, timestep=0.01, total_time=10, Dw=1e-9, kernel=kernel,
                     kernel_derivative=kernel_derivative, pressure_coefficient=pressure_coefficient,
                     viscosity_coefficient=viscosity_coefficient, rest_density=rest_density, gravity=gravity)

    sim.run()


if __name__ == "__main__":
    main()
