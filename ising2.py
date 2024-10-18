import numpy as np
import time_scales as ts
import matplotlib.pyplot as plt
import random
import utils
import time



def default_temp(global_temp, size = 84):
    return np.ones(size) * global_temp


def default_Jij(size = 84):
    return np.fill_diagonal(np.ones((size, size)), 0)


class Spins:

    def __init__(self, Jij, spin_matrix = None):
        self.Jij = utils.normalize_array(Jij)
        self.size = np.shape(Jij)[0]
        if spin_matrix is None:
            self.spins = np.random.choice([-1, 1], self.size)
        else:
            self.spins = spin_matrix
        self.total_energy = self.hamiltonian()
        self.mag = self.magnetization()

    def energy(self, i, j):
        return -2 * np.sum(self.Jij[i, j] * self.spins[i] * self.spins[j])

    def magnetization(self):
        self.mag = np.abs(np.sum(self.spins)) / self.size
        return self.mag

    def find_dE(self, index):
        temp_Jij = self.Jij[index]
        if isinstance(index, list):
            temp_Jij[:, index] = 0
            return -2 * np.sum(temp_Jij * (self.spins * -self.spins[index].reshape((np.size(index), 1))))
        else:
            temp_Jij[index] = 0
            return -2 * np.sum(temp_Jij * (self.spins * -self.spins[index]))

    def hamiltonian(self):
        return np.sum(self.Jij * (self.spins * self.spins.reshape((np.size(self.spins), 1))))

    def update(self, index, energy = None):
        if energy is None:
            energy = self.find_dE(index)
        self.total_energy += energy
        self.spins[index] *= -1


class Ising:

    def __init__(self, spin, temp, steps, thermalization, time_scale = ts.default_time_scale):
        self.spin = spin
        self.iterations = np.arange(steps + 1)
        self.temp = temp
        self.time_scale = time_scale
        self.spin_series = np.zeros((self.spin.size, steps + 1))
        self.spin_series[:, 0] = self.spin.spins
        self.sum_spin_series = np.zeros(steps + 1)
        self.sum_spin_series[0] = np.sum(self.spin.spins)
        self.energy_series = np.zeros(steps + 1)
        self.energy_series[0] = self.spin.total_energy
        self.mag_series = np.zeros(steps + 1)
        self.mag_series[0] = self.spin.magnetization()
        self.thermalization = thermalization
        self.metropolis()
        self.functional_connectivity = self.generate_FC()


    def metropolis(self):
        def metropolis_step(index, dE = None):
            if dE is None:
                dE = self.spin.find_dE(index)
            if dE < 0 or random.random() < np.exp(-dE / self.temp[index]):
                return dE
            else:
                return None

        timer = 0
        for i in self.iterations[1:]:
            start = time.time()
            self.time_scale(metropolis_step, self.spin)
            timer += time.time() - start
            self.spin_series[:, i] = self.spin.spins
            self.sum_spin_series[i] = np.sum(self.spin.spins)
            self.energy_series[i] = self.spin.total_energy
            self.mag_series[i] = self.spin.magnetization()
        print('algorithm time:', timer)

    def generate_FC(self):
        self.functional_connectivity = np.corrcoef(self.spin_series[:, self.thermalization:])
        return self.functional_connectivity

    def susceptibility(self, beta):
        return (np.var(self.mag_series) * self.spin.size ** 2) * beta

    def specific_heat(self, beta):
        return (np.var(self.energy_series) * self.spin.size ** 2) * beta ** 2

    def average_series(self, series):
        return np.cumsum(series) / (self.iterations + 1)

    def series_deriv(self, series):
        return series[:-1] - series[1:]

    def st_dev(self):
        return np.std(self.average_series(self.energy_series)[self.thermalization:])



if __name__ == '__main__':
    folder_path = utils.get_folder('Jij_data')
    matrix_ar = utils.matrix_from_dir(folder_path)
    avg_Jij = utils.average_matrices(*matrix_ar)
    avg_Jij = utils.normalize_array(avg_Jij)
    temp = default_temp(0.7)
    spin = Spins(avg_Jij)
    TS = Ising(spin, temp, 1000)

    figure, axis1 = plt.subplots(1, 2)
    axis1[0].scatter(TS.iterations, TS.energy_series)
    axis1[0].scatter(TS.iterations, TS.average_series(TS.energy_series))
    axis1[1].scatter(TS.iterations, TS.sum_spin_series)
    axis1[1].scatter(TS.iterations, TS.average_series(TS.sum_spin_series))

    figure, axis2 = plt.subplots(1, 2)
    axis2[0].matshow(TS.generate_FC(300))
    axis2[1].matshow(avg_Jij)

    plt.show()
