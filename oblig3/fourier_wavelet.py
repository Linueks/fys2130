from __future__ import division, print_function
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
style.use('ggplot')


# mostly for fourier
sample_rate = 10000
signal_points = 2**13
half = int(signal_points / 2)
total_time = 1.0
time = np.linspace(0, total_time, signal_points)
dt = total_time / signal_points


# mostly for wavelet
min_freq = 800
max_freq = 2000
wave_number = 100
wavelet_points = 1000 #this one was used for the creation of the plots
#wavelet_frequency_points = int(1 + np.log10(max_freq / min_freq) / np.log10(1 + 1 / wave_number)) # 'optimal' number of frequencies
#print(wavelet_frequency_points)

def signal(t):
    f_1 = 1000; f_2 = 1600; c_1 = 1.0; c_2 = 1.7
    value = c_1 * np.sin(2 * np.pi * f_1 * t) + c_2 * np.sin(2 * np.pi * f_2 * t)


    return value


def signal_2(t):
    f_1 = 1000; f_2 = 1600; c_1 = 1.0; c_2 = 1.7;
    t_1 = 0.15; t_2 = 0.5; sig_1 = 0.01; sig_2 = 0.10
    value = c_1 * np.sin(2 * np.pi * f_1 * t) * np.exp(-((t - t_1) / sig_1)**2) + c_2 * np.sin(2 * np.pi * f_2 * t) * np.exp(-((t - t_2) / sig_2)**2)


    return value


def fourier_analysis():
    fourier_transform = np.fft.fft(signal_2(time)) / signal_points
    frequency_range = np.fft.fftfreq(signal_points, dt)


    return fourier_transform, frequency_range



def morlet_wavelet_analytic_fourier(target_frequency, frequency, wave_number):
    C = 0.798 * target_frequency / (sample_rate * wave_number)
    one = C * np.exp(-(wave_number * (frequency - target_frequency) / float(target_frequency))**2)
    two = -C * np.exp(-wave_number**2) * np.exp(-(wave_number * frequency / float(target_frequency))**2)

    return one+two


def wavelet_analysis():
    fourier_coeffs, fourier_frequencies = fourier_analysis()
    frequency_range = 2 * np.pi * fourier_frequencies


    wavelet_frequencies = 2 * np.pi * np.linspace(min_freq, max_freq, wavelet_points)
    wavelet_diagram = np.zeros((wavelet_points, signal_points), dtype=complex)


    for i in range(wavelet_points):
        wavelet_diagram[i] = np.fft.ifft(
                morlet_wavelet_analytic_fourier(wavelet_frequencies[i], frequency_range, wave_number) * fourier_coeffs * signal_points)


    x, y = np.meshgrid(time, np.log10(wavelet_frequencies))


    ax1 = plt.subplot(2, 1, 1, axisbg='white')
    plt.title('Original Signal and Fourier Transform')
    ax1.plot(time, signal_2(time), c='darkblue')
    plt.xlim(0, 0.85)
    plt.xlabel('time [s]')
    plt.ylabel('f(t) [amplitude]')
    ax1.tick_params(
        right = 'off',
        top = 'off'
    )


    ax2 = plt.subplot(2, 1, 2, axisbg='white')
    ax2.plot(fourier_frequencies[:half], np.abs(fourier_coeffs[:half]), c='indianred')
    plt.xlim(0, 2000)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('|X(f)| Fourier Coefficients')
    ax2.tick_params(
        right = 'off',
        top = 'off'
    )

    plt.subplots_adjust(hspace=0.5)

    plt.show()


    #plt.savefig('fourier_transform_gaussian100.png')


    ax3 = plt.subplot(1, 1, 1, axisbg='white')
    plt.title('Wavelet Transform of the Signal')
    plt.pcolormesh(x, y, np.abs(wavelet_diagram), cmap='Blues')
    plt.colorbar()
    plt.xlim(0, 0.85)
    plt.xlabel('time [s]')
    plt.ylabel('log10(frequency) [Hz]')
    ax3.tick_params(
        right = 'off',
        top = 'off'
    )

    plt.show()
    #plt.savefig('wavelet_transform_gaussian100.png')



wavelet_analysis()
