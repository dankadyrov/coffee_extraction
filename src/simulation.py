from typing import Callable, List
from matplotlib import pyplot as plt
from sph_particle import SPHParticle
from celluloid import Camera
import numpy as np
from tqdm import tqdm


class Simulation:
    def __init__(self, particles: List[SPHParticle], timestep: float, total_time: float,
                 Dw: float, kernel: Callable, kernel_derivative: Callable, pressure_coefficient: float,
                 viscosity_coefficient: float, coffee_water_viscosity: float, rest_density: float, gravity: float):
        self.particles = particles
        self.timestep = timestep
        self.total_time = total_time
        self.current_time = 0
        self.Dw = Dw
        self.kernel = kernel
        self.kernel_derivative = kernel_derivative
        self.pressure_coefficient = pressure_coefficient
        self.viscosity_coefficient = viscosity_coefficient
        self.coffee_water_viscosity = coffee_water_viscosity
        self.rest_density = rest_density
        self.gravity = gravity
        self.df = {'time': [], 'mean_c_w': []}

    def run(self) -> None:
        plt.style.use('_mpl-gallery-nogrid')
        fig = plt.figure(figsize=(12, 12))
        camera = Camera(fig)
        with tqdm(total=self.total_time, desc="Progress") as pbar:
            while self.current_time < self.total_time:
                self.update()
                if int(round(self.current_time, 2) * 250) % 10 == 0:
                    self.visualize(self.current_time, camera)
                self.current_time += self.timestep
                pbar.update(self.timestep)
        animation = camera.animate()
        animation.save('../results/model.gif', writer='pillow')

    def update(self) -> None:
        self.compute_densities()
        self.compute_pressure()
        self.compute_c_w(self.timestep)
        forces = self.compute_forces()
        self.apply_forces_and_update(forces)
        mean_c_w = self.calculate_mean_c_w()
        self.df['time'].append(self.current_time)
        self.df['mean_c_w'].append(mean_c_w)

    def apply_forces_and_update(self, forces: List[np.ndarray]) -> None:
        for particle, force in zip(self.particles, forces):
            particle.update(self.timestep, force)

    def find_neighbours(self, particle: SPHParticle, radius: float = 5) -> List[SPHParticle]:
        neighbors = []
        for other in self.particles:
            if np.linalg.norm(particle.position - other.position) < radius:
                neighbors.append(other)
        return neighbors

    '''
    Тут осуществяется рассчёт диффузии по формуле:
    c_1  ->  c_1 - jSt/(V * n) = c_1 - 3/2 * jt/(n * diameter) = c_1 - 3/2 * D * (c_1 - c_2) * t / (diameter * distance)
    Рассчёты проводятся когда "поток" положительный (просто чтобы оно не считалось дважды)
    Диаметр и радиус действия подогнан (радиус действия чтобы на картинке выглядело реалистично)
    '''

    def compute_c_w(self, dt: float, diameter: float = 0.001):
        for particle in self.particles:
            neighbors = self.find_neighbours(particle, 1)
            for neighbor in neighbors:
                distance = np.linalg.norm(particle.position - neighbor.position)
                if not (particle.type == neighbor.type == 'coffee') and particle.c_w > neighbor.c_w:
                    flow = 3 / 2 * self.Dw * (particle.c_w - neighbor.c_w) * dt / (diameter * distance)
                    particle.c_w -= flow
                    neighbor.c_w += flow

    def compute_densities(self) -> None:
        for particle in self.particles:
            density = 0
            neighbors = self.find_neighbours(particle)
            for neighbor in neighbors:
                distance = np.linalg.norm(particle.position - neighbor.position)
                density += neighbor.mass * self.kernel(distance)
            particle.density = density

    def compute_pressure(self) -> None:
        for particle in self.particles:
            particle.pressure = self.pressure_coefficient * (particle.density - self.rest_density)

    def normal(self, distance: float, mass: float, position: np.ndarray) -> np.ndarray:
        h = 0.5
        q = distance / h
        h_squared = h ** 2
        h_9 = h ** 9
        sigma = mass * 315 / (64 * np.pi * h_9)
        if q <= 1:
            return sigma * 3 * (h_squared - distance * distance) ** 2 * 2 * position
        else:
            return np.zeros(2)

    def surface_coef(self, distance: float, mass: float, normal: np.ndarray) -> float:
        h = 0.5
        q = distance / h
        h_squared = h ** 2
        h_9 = h ** 9
        sigma = mass * 315 / (64 * np.pi * h_9)
        d_squared = distance * distance
        module = np.linalg.norm(normal)
        if q <= 1 and module > 0.0001:
            k = -1 * sigma / module * (
                        3 * (h_squared - d_squared) ** 2 * 2 + 3 * (h_squared - d_squared) ** 2 * 2 * d_squared)
            return k
        else:
            return 0

    def compute_forces(self) -> List[np.ndarray]:
        forces = []
        for particle in self.particles:
            force = np.array([0.0, 0.0])
            if particle.type == 'water':
                for neighbor in self.find_neighbours(particle):
                    distance = np.linalg.norm(particle.position - neighbor.position)
                    x, y = particle.position[0] - neighbor.position[0], particle.position[1] - neighbor.position[1]
                    if distance > 0:
                        pressure_force = -1.0 * self.kernel_derivative(x, y) * particle.mass * neighbor.mass * (
                                particle.pressure / particle.density ** 2 + neighbor.pressure / neighbor.density ** 2)
                        viscosity = self.viscosity_coefficient if neighbor.type == particle.type == 'water' else self.coffee_water_viscosity
                        viscosity_force = viscosity * neighbor.mass * (
                                neighbor.velocity - particle.velocity) / neighbor.density * self.kernel(
                            distance)
                        normal = self.normal(distance, particle.mass, particle.position)
                        k = self.surface_coef(distance, particle.mass, normal)
                        tension_force = -0.05 * k * normal
                        force += pressure_force + viscosity_force + tension_force
                force[1] -= self.gravity * self.timestep
            forces.append(force)
        return forces

    def visualize(self, current_time: float, camera: Camera) -> None:
        x_coffee = [p.position[0] for p in self.particles if p.type == 'coffee']
        y_coffee = [p.position[1] for p in self.particles if p.type == 'coffee']
        c_w_coffee = [round(p.c_w, 2) * 100 for p in self.particles if p.type == 'coffee']
        x_water = [p.position[0] for p in self.particles if p.type == 'water']
        y_water = [p.position[1] for p in self.particles if p.type == 'water']
        c_w_water = [round(p.c_w, 2) * 100 for p in self.particles if p.type == 'water']
        plt.scatter(x_coffee, y_coffee, c=c_w_coffee, label='Coffee', alpha=1, s=12, cmap='tab20b')
        plt.scatter(x_water, y_water, c=c_w_water, label='Water', alpha=1, s=2.5, cmap='cool')
        plt.title(f"Current time: {current_time:.2f}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.legend(['Coffee', 'Water'])
        plt.xlim(-20, 20)
        plt.ylim(-20, 20)
        plt.grid(False)
        camera.snap()

    def calculate_mean_c_w(self) -> float:
        mean_c_w = np.array([round(p.c_w, 2) * 100 for p in self.particles if p.type == 'water']).mean()
        return mean_c_w

    def show_mean_concentration_per_time(self) -> None:
        plt.close()
        plt.figure(figsize=(12, 12))
        plt.grid()
        plt.scatter(self.df['time'], self.df['mean_c_w'], s=5)
        plt.xlabel('Time')
        plt.ylabel('Concentration, %')
        plt.tight_layout()
        plt.savefig('../results/concentration.png', dpi=100)
        plt.show()
