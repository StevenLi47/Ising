import numpy as np
import utils
import time


avg_Jij = utils.get_matrix('avg_Jij_no_outliers')
index = None


def default_time_scale(metropolis_step, spins):
    global index
    if index is None:
        index = range(84)

    update_index = []
    for i in index:
        if metropolis_step(i) is not None:
            update_index.append(i)
    spins.update(update_index)


def time_scale1(metropolis_step, spins):
    global index
    index = np.random.randint(84)
    dE = metropolis_step(index)
    if dE is not None:
        spins.update(index, dE)


def time_scale2(metropolis_step, spins):
    global index
    if index is None:
        index = np.zeros(84, dtype = np.uint8)
        avg_connectome = np.mean(avg_Jij, 0)
        sorted_connectome = avg_connectome.copy()
        sorted_connectome.sort()
        sorted_connectome = sorted_connectome[::-1]
        for i in range(84):
            index[i] = np.where(avg_connectome == sorted_connectome[i])[0][0]

    for i in index:
        dE = metropolis_step(i)
        if dE is not None:
            spins.update(i, dE)
