import numpy as np
import matplotlib.pyplot as plt
import random


def default_time_scale(metropolis_step, spins):
    index = []
    for i in range(84):
        if metropolis_step(i):
            index.append(i)

    spins.update(index)


def default_params(size, global_temp):
    temp = np.ones(size) * global_temp
    Jij = np.ones((size, size)) - np.diag(np.ones(size))
    spins = Spins(np.random.choice([-1, 1], size), Jij)

    return temp, spins


class Spins:

    def __init__(self, spin_matrix, Jij):
        self.spins = spin_matrix
        self.size = np.size(spin_matrix)
        self.Jij = Jij
        self.total_energy = self.hamiltonian()

    def energy(self, i, j, flip = False):
        if flip:
            return np.sum(self.Jij[i, j] * -self.spins[i] * self.spins[j])
        else:
            return np.sum(self.Jij[i, j] * self.spins[i] * self.spins[j])

    def hamiltonian(self):
        energy = 0

        for i in range(self.size - 1):
            energy -= np.sum(self.spins[i] * self.spins[i + 1:] * self.Jij[i, i + 1:])

        return energy

    def update(self, index):
        self.spins[index] *= -1
        self.total_energy = self.hamiltonian()


class Ising:

    def __init__(self, spin, temp, steps, time_scale = default_time_scale):
        self.spin = spin
        self.iterations = np.arange(steps)
        self.temp = temp
        self.time_scale = time_scale
        self.spin_series = np.zeros((self.spin.size, steps))
        self.sum_spin_series = np.zeros(steps)
        self.energy_series = np.zeros(steps)
        self.metropolis()

    def metropolis(self):
        def metropolis_step(index):
            dE = 2 * self.spin.energy(index, range(self.spin.size))

            if dE < 0 or random.random() < np.exp(-dE / temp[index]):
                return True
            else:
                return False

        for i in self.iterations:
            self.time_scale(metropolis_step, self.spin)
            self.spin_series[:, i] = self.spin.spins
            self.sum_spin_series[i] = np.sum(self.spin.spins)
            self.energy_series[i] = self.spin.total_energy

    def generate_FC(self, thermalization):
        self.functional_connectivity = np.corrcoef(self.spin_series[:, thermalization:])
        return self.functional_connectivity

    def average_series(self, series):
        return np.cumsum(series) / (self.iterations + 1)

    def spin_plot(self):
        plot = plt.figure()
        plt.scatter(self.iterations, self.sum_spin_series)
        plt.scatter(self.iterations, self.average_series(self.sum_spin_series))
        return plot

    def energy_plot(self):
        plot = plt.figure()
        plt.scatter(self.iterations, self.energy_series)
        plt.scatter(self.iterations, self.average_series(self.energy_series))
        return plot


temp, spins = default_params(84, 60)
TS = Ising(spins, temp, 3000)
plot = TS.energy_plot()
print(TS.generate_FC(1500))
plt.show()
