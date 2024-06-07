'''
Переписала ядро и его градиент, скоректировала взаимодействие между всеми частицами (вода-вода и вода-кофе). Изменила
силы, внесла изменения в нахождение соседей, добавила вязкость внутри кофе (чтобы вода на скорости не пролетала).
На данный момент проблема, что молекулы воды не затекают внутрь кофе (видимо потому что там повышенное давление, не знаю),
И вода собирается в пузыри
'''

import numpy as np
from matplotlib import pyplot as plt
from typing import Callable, List


# Тут конечно захардкодил, но пока пойдет
# Кофе должен иметь некоторую стуртуру, поэтому было переписано создание кофейной частицы. Теперь кофе имеет ту же структуру, что и в статье
def coffee_visualizer(mass: float) -> List:
    step = 0.75
    x_min, x_max = -5, 5
    y_min, y_max = -5, 5
    coffee_particles = []

    for i in range(11):
        for j in range(11):
            x = x_min + j * step
            y = y_min + i * step
            coffee_particles.append(SPHParticle(np.array([x, y]), [0.0, 0.0], mass, 'coffee'))

    current_x = x_min + step * 11
    current_y = y_min + step
    for i in range(9):
        y = current_y + i * step
        coffee_particles.append(SPHParticle(np.array([current_x, y]), [0.0, 0.0], mass, 'coffee'))

    current_x = x_min - step
    for i in range(9):
        y = current_y + i * step
        coffee_particles.append(SPHParticle(np.array([current_x, y]), [0.0, 0.0], mass, 'coffee'))

    current_x = x_min + step
    current_y = y_min - step
    for i in range(9):
        x = current_x + i * step
        coffee_particles.append(SPHParticle(np.array([x, current_y]), [0.0, 0.0], mass, 'coffee'))

    current_y = y_min + step * 11
    for i in range(9):
        x = current_x + i * step
        coffee_particles.append(SPHParticle(np.array([x, current_y]), [0.0, 0.0], mass, 'coffee'))

    current_x = x_min + step * 12
    current_y = y_min + step * 3
    for i in range(5):
        y = current_y + i * step
        coffee_particles.append(SPHParticle(np.array([current_x, y]), [0.0, 0.0], mass, 'coffee'))

    current_x = x_min - 2 * step
    for i in range(5):
        y = current_y + i * step
        coffee_particles.append(SPHParticle(np.array([current_x, y]), [0.0, 0.0], mass, 'coffee'))

    current_x = x_min + 3 * step
    current_y = y_min - 2 * step
    for i in range(5):
        x = current_x + i * step
        coffee_particles.append(SPHParticle(np.array([x, current_y]), [0.0, 0.0], mass, 'coffee'))

    current_y = y_min + step * 12
    for i in range(5):
        x = current_x + i * step
        coffee_particles.append(SPHParticle(np.array([x, current_y]), [0.0, 0.0], mass, 'coffee'))

    return coffee_particles


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

    '''def update_position(self, dt: float) -> None:
        self.position += self.velocity * dt'''

    '''def apply_force(self, force: np.ndarray, dt: float) -> None:
        acceleration = force / self.mass
        self.velocity += acceleration * dt'''

    '''def update_c_w(self, Dw: float, dt: float, neighbors: List['SPHParticle']):
        for neighbor in neighbors:
            if neighbor.type == 'water':
                distance = np.linalg.norm(self.position - neighbor.position)
                self.c_w += Dw * (neighbor.c_w - self.c_w) / (distance ** 2) * dt'''


class CoffeeGrain:
    def __init__(self, particles: List[SPHParticle]):
        self.particles = particles

    '''def update_particles(self, dt: float) -> None:
        for particle in self.particles:
            particle.update_position(dt)'''

    '''def apply_forces(self, forces: List[np.ndarray], dt: float) -> None:
        for particle, force in zip(self.particles, forces):
            particle.apply_force(force, dt)'''

    '''def update_swelling(self, Dw: float, dt: float) -> None:
        for particle in self.particles:
            neighbors = self.find_neighbours(particle)
            # particle.update_c_w(Dw, dt, neighbors)
            swelling_factor = (1 - particle.c_w) ** (-1 / 3)
            particle.position *= swelling_factor'''

    def find_neighbours(self, particle: SPHParticle, radius: float = 1.0) -> List[SPHParticle]:
        neighbors = []
        for other in self.particles:
            if np.linalg.norm(particle.position - other.position) < radius:
                neighbors.append(other)
        return neighbors


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
        self.coffee_water_viscosity = coffee_water_viscosity  # добавила коэффициент вязкости внутри кофе
        self.rest_density = rest_density
        self.gravity = gravity

    def run(self) -> None:
        while self.current_time < self.total_time:
            self.update()
            if int(round(self.current_time, 2) * 250) % 10 == 0:  # чаще обновления экрана
                self.visualize(self.current_time)
            self.current_time += self.timestep

    def update(self) -> None:
        self.compute_densities()
        self.compute_pressure()
        forces = self.compute_forces()
        self.apply_forces_and_update(forces)
        # self.grain.update_swelling(self.Dw, self.timestep)  # убрала учёт поглощения

    def apply_forces_and_update(self, forces: List[np.ndarray]) -> None:
        for particle, force in zip(self.particles, forces):
            particle.update(self.timestep, force)
        '''self.grain.apply_forces(forces, self.timestep)
        self.grain.update_particles(self.timestep)'''

    # сложность всё равно n^2, так что можно и без этого делать (с соседями даже медленнее на константу)
    def find_neighbours(self, particle: SPHParticle, radius: float = 5) -> List[
        SPHParticle]:  # поиск соседей для любых частиц
        neighbors = []
        for other in self.particles:
            if np.linalg.norm(particle.position - other.position) < radius:
                neighbors.append(other)
        return neighbors

    def compute_densities(self) -> None:
        for particle in self.particles:  # заменила молекулы кофе на все
            density = 0
            neighbors = self.find_neighbours(particle)  # ищу соседей для любой частицы
            for neighbor in neighbors:
                distance = np.linalg.norm(particle.position - neighbor.position)
                density += neighbor.mass * self.kernel(distance)  # поменяла формулу в соответствии с 1 файлом
            particle.density = density

    def compute_pressure(self) -> None:
        for particle in self.particles:  # заменила частицы кофе на все частицы
            particle.pressure = self.pressure_coefficient * (particle.density - self.rest_density)

    def normal(self, distance: float, mass: float, position: np.ndarray) -> np.ndarray:
        h = 1
        q = distance / h
        h_squared = h ** 2
        h_9 = h ** 9
        sigma = mass * 315 / (64 * np.pi * h_9)
        if q <= 1:
            return sigma * 3 * (h_squared - distance * distance) ** 2 * 2 * position
        else:
            return np.zeros(2)

    def surface_coef(self, distance: float, mass: float, normal: np.ndarray) -> float:
        h = 0.25
        q = distance / h
        h_squared = h ** 2
        h_9 = h ** 9
        sigma = mass * 315 / (64 * np.pi * h_9)
        d_squared = distance * distance
        if q <= 1:
            module = np.linalg.norm(normal)
            k = -1 * sigma / module * (3 * (h_squared - d_squared) ** 2 * 2 + 3 * (h_squared - d_squared) ** 2 * 2 * d_squared)
            return k
        else:
            return 0

    def compute_forces(self) -> List[np.ndarray]:
        forces = []
        for particle in self.particles:
            force = np.array([0.0, 0.0])
            if particle.type == 'water':  # делаем рассчёт сил только для частиц воды
                for neighbor in self.find_neighbours(particle):
                    distance = np.linalg.norm(particle.position - neighbor.position)
                    x, y = particle.position[0] - neighbor.position[0], particle.position[1] - neighbor.position[
                        1]  # координаты вектора между 2 молекулами, нужны для
                    if distance > 0:
                        pressure_force = -1.0 * self.kernel_derivative(x, y) * particle.mass * neighbor.mass * (
                                particle.pressure / particle.density ** 2 + neighbor.pressure / neighbor.density ** 2)  # поменяла формулу для силы
                        viscosity = self.viscosity_coefficient if neighbor.type == particle.type == 'water' else self.coffee_water_viscosity
                        viscosity_force = viscosity * neighbor.mass * (
                                neighbor.velocity - particle.velocity) / neighbor.density * self.kernel(
                            distance)  # поменяла формулу для силы
                        normal = self.normal(distance, particle.mass, particle.position)
                        k = self.surface_coef(distance, particle.mass, normal)
                        tension_force = -0.05 * k * normal
                        force += pressure_force + viscosity_force + tension_force
                force[1] -= self.gravity * self.timestep
            forces.append(force)
        return forces

    def visualize(self, current_time: float) -> None:
        plt.close()  # чтобы график закрывался только перед новой отрисовкой
        x_coffee = [p.position[0] for p in self.particles if
                    p.type == 'coffee']  # поменяла массив grain.particles -> particles
        y_coffee = [p.position[1] for p in self.particles if p.type == 'coffee']
        x_water = [p.position[0] for p in self.particles if p.type == 'water']
        y_water = [p.position[1] for p in self.particles if p.type == 'water']
        plt.figure(figsize=(12, 12))
        plt.scatter(x_coffee, y_coffee, color='brown', label='Coffee', alpha=1, s=12)
        plt.scatter(x_water, y_water, color='blue', label='Water', alpha=0.5, s=2.5)
        plt.title(f"Current time: {current_time:.2f}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.xlim(-20, 20)
        plt.ylim(-20, 20)
        plt.grid(False)
        plt.show(block=False)
        plt.pause(0.1)
        # plt.close()


def main():
    h = 0.25 # Другой размер ядра, пока что оно лучше всего работает
    pressure_coefficient = 10.0
    viscosity_coefficient = 10.0
    coffee_water_viscosity = 500.0
    gravity = 1000.0
    rest_density = 1.0

    def kernel(distance) -> float:  # переписала функцию в соответствии со статьей
        q = distance / h
        sigma = 7 / (478 * np.pi * h ** 2)
        if q <= 1:
            return sigma * ((3 - q) ** 5 - 6 * (2 - q) ** 5 + 15 * (1 - q) ** 5)
        elif q <= 2:
            return sigma * ((3 - q) ** 5 - 6 * (2 - q) ** 5)
        elif q <= 3:
            return sigma * (3 - q) ** 5
        else:
            return 0

    def kernel_derivative(x, y) -> np.ndarray[
        float, float]:  # переписала функцию в соответствии с предыдущей + чтобы она возвращала двумерный массив
        q = np.sqrt(x ** 2 + y ** 2) / h
        dq_dx = x / (h * np.sqrt(x ** 2 + y ** 2))
        dq_dy = y / (h * np.sqrt(x ** 2 + y ** 2))
        sigma = 7 / (478 * np.pi * h ** 2)
        if q <= 1:
            kernel_der_x = sigma * (- 5 * (3 - q) ** 4 + 30 * (2 - q) ** 4 - 75 * (1 - q) ** 4) * dq_dx
            kernel_der_y = sigma * (- 5 * (3 - q) ** 4 + 30 * (2 - q) ** 4 - 75 * (1 - q) ** 4) * dq_dy
        elif q <= 2:
            kernel_der_x = sigma * (- 5 * (3 - q) ** 4 + 30 * (2 - q) ** 4) * dq_dx
            kernel_der_y = sigma * (- 5 * (3 - q) ** 4 + 30 * (2 - q) ** 4) * dq_dy
        elif q <= 3:
            kernel_der_x = sigma * (- 5 * (3 - q) ** 4) * dq_dx
            kernel_der_y = sigma * (- 5 * (3 - q) ** 4) * dq_dy
        else:
            return np.array([0, 0])
        return np.array([kernel_der_x, kernel_der_y])

    # Генерируем точки воды в прямоугольнике с равным шагом
    x_min, x_max = -10, 10
    y_min, y_max = 5, 15

    step = 0.75
    num_points_x = int((x_max - x_min) / step) + 1
    num_points_y = int((y_max - y_min) / step) + 1
    water_particles = []
    for i in range(num_points_y):
        for j in range(num_points_x):
            x = x_min + j * step
            y = y_min + i * step
            print(x, y)
            water_particles.append(SPHParticle(np.array([x, y]), [0.0, 1.0], 1.0, 'water'))

    print(len(water_particles))

    # grain = CoffeeGrain(coffee_particles)
    particles = coffee_visualizer(1.25)
    particles = list(np.append(particles, water_particles))  # создала один массив со всеми частицами

    sim = Simulation(particles, timestep=0.01, total_time=10, Dw=1e-9, kernel=kernel,
                     kernel_derivative=kernel_derivative, pressure_coefficient=pressure_coefficient,
                     viscosity_coefficient=viscosity_coefficient, coffee_water_viscosity=coffee_water_viscosity,
                     rest_density=rest_density, gravity=gravity)  # поменяла подаваемые параметры

    sim.run()


if __name__ == "__main__":
    main()
