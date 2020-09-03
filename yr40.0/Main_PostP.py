#!/usr/bin/env python
import Functions_PostP as Fun
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

file_names = np.array(pd.read_csv('1metadata.csv', header=None))
heat_map_axes = np.genfromtxt('0metadata.csv', delimiter=',')
heat_map_axes = np.round(heat_map_axes, 2)
heat_map_axis1 = heat_map_axes[0, 0:9]
heat_map_axis2 = heat_map_axes[1, :]
heat_map_matrices = np.zeros([4, 9, file_names.shape[1]])

plot0 = Fun.burn_in_time_series(np.genfromtxt(file_names[7, 0], delimiter=','), 100)
plot1 = Fun.low_pass_filter(Fun.uniformly_sample(Fun.burn_in_time_series(np.genfromtxt(file_names[7, 0],
                                                                                       delimiter=','),
                                                                         100), 10000),
                            [1/256, 1/32, 7/64, 7/32, 35/128, 7/32, 7/64, 1/32, 1/256])
plot2 = Fun.low_pass_filter(Fun.uniformly_sample(Fun.burn_in_time_series(np.genfromtxt(file_names[7, 0],
                                                                                       delimiter=','),
                                                                         100), 5000),
                            [1/256, 1/32, 7/64, 7/32, 35/128, 7/32, 7/64, 1/32, 1/256])
plot3 = Fun.low_pass_filter(Fun.uniformly_sample(Fun.burn_in_time_series(np.genfromtxt(file_names[7, 0],
                                                                                       delimiter=','),
                                                                         100), 1000),
                            [1/256, 1/32, 7/64, 7/32, 35/128, 7/32, 7/64, 1/32, 1/256])
plot4 = Fun.low_pass_filter(Fun.uniformly_sample(Fun.burn_in_time_series(np.genfromtxt(file_names[7, 0],
                                                                                       delimiter=','),
                                                                         100), 100),
                            [1/256, 1/32, 7/64, 7/32, 35/128, 7/32, 7/64, 1/32, 1/256])
plot5 = Fun.low_pass_filter(Fun.uniformly_sample(Fun.burn_in_time_series(np.genfromtxt(file_names[7, 0],
                                                                                       delimiter=','),
                                                                         100), 14),
                            [1/256, 1/32, 7/64, 7/32, 35/128, 7/32, 7/64, 1/32, 1/256])
plt.plot(plot0[:, 0] + 100, plot0[:, 1])
plt.show()
plt.plot(plot1[:, 0] + 100, plot1[:, 1])
plt.show()
plt.plot(plot2[:, 0] + 100, plot2[:, 1])
plt.show()
plt.plot(plot3[:, 0] + 100, plot3[:, 1])
plt.show()
plt.plot(plot4[:, 0] + 100, plot4[:, 1])
plt.show()
plt.plot(plot5[:, 0] + 100, plot5[:, 1])
plt.show()

for mean_axis in range(9):
    for stdev_axis in range(file_names.shape[1]):
        t1 = time.time()
        stats = Fun.all_together_now(np.genfromtxt(file_names[mean_axis, stdev_axis], delimiter=','),
                                     3000, 100, [1/256, 1/32, 7/64, 7/32, 35/128, 7/32, 7/64, 1/32, 1/256])
        heat_map_matrices[:, mean_axis, stdev_axis] = stats
        time.sleep(1)
        t2 = time.time()
        print(t2 - t1)

mean_period = pd.DataFrame(heat_map_matrices[0, :, :], index=heat_map_axis1, columns=heat_map_axis2)
mean_amplitude = pd.DataFrame(heat_map_matrices[1, :, :], index=heat_map_axis1,
                              columns=heat_map_axis2)
period_CV = pd.DataFrame(heat_map_matrices[2, :, :], index=heat_map_axis1, columns=heat_map_axis2)
amplitude_CV = pd.DataFrame(heat_map_matrices[3, :, :], index=heat_map_axis1, columns=heat_map_axis2)

mean_period_heat_map = Fun.generate_heat_map(mean_period, 'Mean Period', ['CV of Delay', 'Delay Mean (min)'],
                                             np.mean(list(heat_map_matrices[0, :, :])))
mean_amplitude_heat_map = Fun.generate_heat_map(mean_amplitude, 'Mean Amplitude', ['CV of Delay', 'Delay Mean (min)'],
                                                np.mean(list(heat_map_matrices[1, :, :])))
period_CV_heat_map = Fun.generate_heat_map(period_CV, 'CV of Period', ['CV of Delay', 'Delay Mean (min)'],
                                           np.mean(list(heat_map_matrices[2, :, :])))
amplitude_CV_heat_map = Fun.generate_heat_map(amplitude_CV, 'CV of Amplitude', ['CV of Delay', 'Delay Mean (min)'],
                                              np.mean(list(heat_map_matrices[3, :, :])))

normalized_heat_map_matricies = np.zeros(heat_map_matrices.shape)
for index0 in range(4):
    for index1 in range(9):
        normalized_heat_map_matricies[index0, index1, :] = heat_map_matrices[index0, index1, :] / \
                                               heat_map_matrices[index0, index1, 0]

normalized_mean_period = pd.DataFrame(normalized_heat_map_matricies[0, :, :],
                                      index=heat_map_axis1, columns=heat_map_axis2)
normalized_mean_amplitude = pd.DataFrame(normalized_heat_map_matricies[1, :, :],
                                         index=heat_map_axis1, columns=heat_map_axis2)
normalized_period_CV = pd.DataFrame(normalized_heat_map_matricies[2, :, :],
                                    index=heat_map_axis1, columns=heat_map_axis2)
normalized_amplitude_CV = pd.DataFrame(normalized_heat_map_matricies[3, :, :],
                                       index=heat_map_axis1, columns=heat_map_axis2)

normalized_mean_period_heat_map = Fun.generate_heat_map(normalized_mean_period,
                                                        'Mean Period Normalized by Delta Delay',
                                                        ['CV of Delay', 'Delay Mean (min)'], 1)
normalized_mean_amplitude_heat_map = Fun.generate_heat_map(normalized_mean_amplitude,
                                                           'Mean Amplitude Normalized by Delta Delay',
                                                           ['CV of Delay', 'Delay Mean (min)'], 1)
normalized_period_CV_heat_map = Fun.generate_heat_map(normalized_period_CV,
                                                      'CV of Period Normalized by Delta Delay',
                                                      ['CV of Delay', 'Delay Mean (min)'], 1)
normalized_amplitude_CV_heat_map = Fun.generate_heat_map(normalized_amplitude_CV,
                                                         'CV of Amplitude Normalized by Delta Delay',
                                                         ['CV of Delay', 'Delay Mean (min)'], 1)
