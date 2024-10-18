import numpy as np
import scipy.stats as sp
import os
import matplotlib.pyplot as plt
import ising2 as I
import utils
import time_scales as ts
import time
from multiprocessing import Pool

#multiplier_array = np.random.choice(np.linspace(0.5, 2, 100), 84)
multiplier_array = np.ones(84)
np.random.seed(0)
steps = 2000
min_temp = 1
max_temp = 3
temp_step_size = 20
thermalization = 800
time_scale = ts.default_time_scale
#spin_array = np.random.choice([-1, 1], 84)
spin_array = np.ones(84)

avg_Jij = utils.get_matrix('avg_Jij_no_outliers')
FC_ar = [utils.get_matrix('avg_TS_1'), utils.get_matrix('avg_TS_2'), utils.get_matrix('avg_TS_3')]
FC_ar.append(utils.average_matrices(*FC_ar))

st_dev_ar = []
ising_ar = []
mag_ar = []
suscept_ar = []
spec_heat_ar = []
corr_ar_1, corr_ar_2, corr_ar_3, corr_ar_total = [], [], [], []
temp_ar = np.linspace(min_temp, max_temp, temp_step_size)

plt.ion()
for t in temp_ar:
    temp = I.default_temp(t) * multiplier_array
    temp[temp > max_temp] = max_temp
    spin = I.Spins(avg_Jij, spin_array.copy())
    TS = I.Ising(spin, temp, steps, thermalization, time_scale = time_scale)
    FC = np.nan_to_num(TS.functional_connectivity)
    ising_ar.append(TS)
    st_dev = TS.st_dev()
    st_dev_ar.append(st_dev)
    mag_ar.append(np.mean(TS.mag_series))
    suscept_ar.append(TS.susceptibility(1/t))
    spec_heat_ar.append(TS.specific_heat(1/t))
    print('temp:', t, ' std:', st_dev)

    #corr_ar_1.append(np.abs(sp.pearsonr(FC.flatten(), FC_ar[0].flatten()))[0])
    #corr_ar_2.append(np.abs(sp.pearsonr(FC.flatten(), FC_ar[1].flatten()))[0])
    #corr_ar_3.append(np.abs(sp.pearsonr(FC.flatten(), FC_ar[2].flatten()))[0])
    #corr_ar_total.append(np.abs(sp.pearsonr(FC.flatten(), FC_ar[3].flatten()))[0])
    corr_ar_1.append(np.abs(sp.pearsonr(utils.flat_remove_diag(FC), utils.flat_remove_diag(FC_ar[0]))[0]))
    corr_ar_2.append(np.abs(sp.pearsonr(utils.flat_remove_diag(FC), utils.flat_remove_diag(FC_ar[1]))[0]))
    corr_ar_3.append(np.abs(sp.pearsonr(utils.flat_remove_diag(FC), utils.flat_remove_diag(FC_ar[2]))[0]))
    corr_ar_total.append(np.abs(sp.pearsonr(utils.flat_remove_diag(FC), utils.flat_remove_diag(FC_ar[3]))[0]))

    if t != temp_ar[0]:
        plt.close()
    figure, axis = plt.subplots(1, 3)
    #axis[0].set_ylim([-90, 90])
    #axis[0].plot(TS.series_deriv(TS.sum_spin_series), color = 'green')
    axis[0].scatter(TS.iterations, TS.mag_series)
    axis[0].plot(TS.iterations, TS.average_series(TS.mag_series), 'r')
    axis[1].set_ylim([np.min(TS.energy_series), np.max(TS.energy_series)])
    axis[1].scatter(TS.iterations, TS.energy_series)
    axis[1].plot(TS.iterations, TS.average_series(TS.energy_series), 'r')
    #axis[1].plot(TS.series_deriv(TS.energy_series), color = 'green')
    axis[2].matshow(TS.functional_connectivity)
    figure.canvas.draw()
    figure.canvas.flush_events()
plt.ioff()

index = np.where(corr_ar_total == np.nanmax(corr_ar_total))[0][0]
best_temp = temp_ar[index]
print('best temp:', best_temp)
for i in range(4):
    print('FC vs Jij:', sp.pearsonr(FC_ar[i].flatten(), avg_Jij.flatten())[0])
figure, axis = plt.subplots(1, 2)
axis[0].matshow(FC_ar[1])
axis[1].matshow(ising_ar[index].functional_connectivity)
plt.show()

figure, axis = plt.subplots(1, 2)
axis[0].plot(temp_ar, spec_heat_ar)
axis[1].plot(temp_ar, corr_ar_1)
axis[1].plot(temp_ar, corr_ar_2)
axis[1].plot(temp_ar, corr_ar_3)
axis[1].plot(temp_ar, corr_ar_total)
plt.show()
