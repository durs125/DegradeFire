import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


def burn_in_time_series(signal, burn_in_time):
    temp_signal = signal
    temp_signal[:, 0] = signal[:, 0] - burn_in_time
    new_start_time = np.where(temp_signal[:, 0] > 0)[0][0]
    new_start_state = np.zeros(temp_signal.shape[1])
    new_start_state[1:] = temp_signal[new_start_time - 1, 1:]
    temp_signal[new_start_time - 1, :] = new_start_state
    burned_in_signal = temp_signal[(new_start_time - 1):, :]
    return burned_in_signal


def uniformly_sample(signal, number_of_samples):
    n = min(np.shape(signal))
    uniform_sampling = np.zeros([number_of_samples, n])
    uniform_timestamps = np.linspace(0, signal[-1, 0], number_of_samples)
    uniform_sampling[:, 0] = uniform_timestamps
    counter = 0
    for index in range(number_of_samples):
        while counter < max(np.shape(signal)):
            if signal[counter, 0] > uniform_timestamps[index]:
                uniform_sampling[index, 1:n] = signal[counter - 1, 1:n]
                break
            counter += 1
    return uniform_sampling


def low_pass_filter(signal, weights):
    if len(weights) % 2 == 1:
        chop = int((len(weights) - 1) / 2)
        filtered_signal = np.zeros([max(signal.shape) - chop * 2, 2])
        weights = np.array(weights)
        for index in range(max(filtered_signal.shape)):
            filtered_signal[index, 0] = signal[index + chop, 0]
            filtered_signal[index, 1] = np.dot(weights, signal[index:(index+chop*2+1), 1])
        return filtered_signal
    else:
        print("Don't make me do this shit.")
        return "error"


def is_max_in_window(signal, length_of_signal, window_size):
    def window_checker(index):
        if index <= window_size or index >= length_of_signal - window_size:
            return np.float16(1 + np.random.uniform())
        elif signal[index, 1] < signal[index - window_size, 1] \
                or signal[index, 1] < signal[index + window_size, 1]:
            return np.float16(1 + np.random.uniform())
        else:
            return np.float16(0)
    return window_checker


def compute_optimal_time_window(signal, splits_matrix_into=1):
    """ first half of the peak detection algorithm """
    n = max(np.shape(signal))  # number of samples
    rows = int(np.ceil(n / 2) - 1)  # length of lms matrix
    lms = np.zeros((rows, n))  # everything is a peak until proven guilty
    for x in range(0, rows):  # I wrote a function that creates a function to be called in another function for this...
        lms[x, :] = np.array(list(map(is_max_in_window(signal, n, x + 1), range(n))))
    row_sum = np.sum(lms, axis=1)
    gamma = np.where(row_sum == np.amin(row_sum))
    rescaled_lms = np.vsplit(lms, gamma[0] + 1)[0]  # cut the lms at the optimal search
    return rescaled_lms


def detect_peaks(signal):
    print('detecting peaks')
    column_sd = np.std(compute_optimal_time_window(signal), axis=0)
    # where columns sum to zero is where local maxima occur
    peaks_index = np.where(column_sd == 0)  # peaks occur when column-wise standard deviation is zero
    peaks = signal[peaks_index, :]  # select peaks based on their index
    peaks = peaks[0, :, :]  # I don't know why this is necessary but it is
    return peaks  # pick off the peaks with timestamps and return them in a numpy array


def run_statistics(peaks):
    print('generating statistics')
    mean_amplitude = np.mean(peaks[:, 1])
    mean_period = np.mean(np.diff(peaks[:, 0]))
    amplitude_coefficient_of_variation = np.std(peaks[:, 1]) / mean_amplitude
    period_coefficient_of_variation = np.std(np.diff(peaks[:, 0])) / mean_period
    return [mean_period, mean_amplitude, period_coefficient_of_variation, amplitude_coefficient_of_variation]


def all_together_now(signal, number_of_samples, burn_in_time, weights):
    print('cleaning signal')
    clean_signal = low_pass_filter(uniformly_sample(burn_in_time_series(signal,
                                                                        burn_in_time),
                                                    number_of_samples),
                                   weights)
    return run_statistics(detect_peaks(clean_signal))


def generate_heat_map(data, title, axis_labels, heat_center):
    heat_map = sb.heatmap(data, center=heat_center)
    heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=0)
    heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=30)
    heat_map.invert_yaxis()
    plt.title(title)
    plt.ylabel(axis_labels[1])
    plt.xlabel(axis_labels[0])
    file_name = title + '.png'
    heat_map.get_figure().savefig(file_name)
    return heat_map


def plot_time_series_with_peaks(signal, burn_in_time, weights):
    burned_in_signal = burn_in_time_series(signal, burn_in_time)
    uniform_signal = uniformly_sample(burned_in_signal, round(max(burned_in_signal.shape)/10))
    clean_signal = low_pass_filter(uniform_signal, weights)
    peaks = detect_peaks(clean_signal)
    curve = plt.plot(clean_signal[0, :], clean_signal[1, :])
    plt.show()
    maxima = sb.scatterplot(peaks[0, :], peaks[1, :])
    plt.show()
    return [curve, maxima]
