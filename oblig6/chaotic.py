from __future__ import division, print_function
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
style.use('ggplot')


## Distribution Parameters
no_points = 4000
freq_center = 5000
freq_full_width = 3000
sample_rate = 8 * (freq_center + freq_full_width)
total_time = no_points / sample_rate

# Wavelet parameters
min_freq = 5000
max_freq = 15000
wave_number = 24
wavelet_points = 1000
#time = np.linspace(0, total_time, no_points)


def morlet_wavelet_analytic_fourier(target_frequency, frequency, wave_number):
    C = 0.798 * target_frequency / (sample_rate * wave_number)
    one = C * np.exp(-(wave_number * (frequency - target_frequency) / float(target_frequency))**2)
    two = -C * np.exp(-wave_number**2) * np.exp(-(wave_number * frequency / float(target_frequency))**2)

    return one+two


def wavelet_analysis(fourier_coeffs, fourier_frequencies, time):


    frequency_range = 2 * np.pi * fourier_frequencies


    wavelet_frequencies = 2 * np.pi * np.linspace(min_freq, max_freq, wavelet_points)
    wavelet_diagram = np.zeros((wavelet_points, len(time)), dtype=complex)


    for i in range(wavelet_points):
        wavelet_diagram[i] = np.fft.ifft(
                morlet_wavelet_analytic_fourier(wavelet_frequencies[i], frequency_range, wave_number) * fourier_coeffs * len(time))


    x, y = np.meshgrid(time, wavelet_frequencies / (2 * np.pi))


    return wavelet_diagram, x, y


def generate_chaotic_signal(Fs, N, f_0, fwhm):
    """
    Function which generates a chaotic signal in the time domain,
    from a gaussian distribution of frequencies.
    """

    freq_standard_deviation = fwhm / 2


    times = np.linspace(0, total_time * (N-1) / N, N)
    frequencies = np.linspace(0, Fs * (N-1) / N, N)


    guassian_envelope = np.exp(-(frequencies-f_0)**2 / (freq_standard_deviation**2))
    amplitudes = np.random.random(N) * np.transpose(guassian_envelope)
    phases = 2 * np.pi * np.random.rand(N)
    frequency_spectrum = amplitudes * (np.cos(phases) + 1j * np.sin(phases))


    ## Speiler nedre del rundt (N_half + 1) for A fAA ovre del korrekt
    n_half = int(np.round(N / 2))


    for i in range(n_half-1):
        frequency_spectrum[N-i-1] = np.conjugate(frequency_spectrum[i+1])

    frequency_spectrum[n_half] = frequency_spectrum[n_half].real
    frequency_spectrum[0] = 0.0

    time_signal = np.real(np.fft.ifft(frequency_spectrum) * 200)

    return time_signal, frequency_spectrum, frequencies, times


def calculate_autocorrelation(Fs, g, time):
    """
    Looks at temporal correlation in a signal g:
    for j = 0, ..., N - M:  (M < N)
        C(j+1) = \frac{\sum_i^M g(i) * g(i + j)}{\sum_i^M g(i) * g(i)}
    """

    #from stackx testing skjonner ikke,,
    #calc = np.array([g[0 : len(g)-time], g[time:]])
    #result = np.corrcoef(g)
    #result = result[result.size/2:]
    #plt.plot(result)
    #plt.show()

    N = int(len(time))
    M = int(N / 2)

    C = np.zeros(N-M)
    for j in range((N-M)-1):
        teller = 0
        nevner = 0
        for i in range(M):
            teller += g[i] * g[i+j]
            nevner += g[i] * g[i]
        C[j] = teller / nevner

    return C


if __name__ == '__main__':


    time_signal, frequency_spectrum, frequencies, time = generate_chaotic_signal(sample_rate, no_points, freq_center, freq_full_width)
    C = calculate_autocorrelation(sample_rate, time_signal, time)

    #"""
    ax1 = plt.subplot(2, 1, 1)
    plt.title('Time Signal g(t)')
    ax1.plot(1000 * time, time_signal, c='darkblue')
    plt.xlabel('time [ms]')
    plt.ylabel('g(t)')
    ax1.tick_params(
        right = 'off',
        top = 'off'
    )


    ax2 = plt.subplot(2, 1, 2)
    plt.title('Fourier Spectre F(g(t))')
    ax2.plot(frequencies, frequency_spectrum)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Fourier Coefficient')
    ax2.tick_params(
        right = 'off',
        top = 'off'
    )

    plt.subplots_adjust(hspace=0.9)

    plt.show()


    ax3 = plt.subplot(1, 1, 1)
    plt.title('Auto-Correlation Function C(T)')
    plt.plot(C)
    plt.xlabel('Time lag [ms]')
    plt.ylabel('Correlation Coefficient')
    ax3.tick_params(
        right = 'off',
        top = 'off'
    )

    plt.show()


    wavelet_diagram, x, y = wavelet_analysis(frequency_spectrum, frequencies, time)

    ax4 = plt.subplot(1, 1, 1, axisbg='white')
    plt.title('Wavelet Transform of the Signal')
    plt.pcolormesh(1000 * x, y, np.abs(wavelet_diagram), cmap='Blues')
    plt.colorbar()
    plt.xlabel('time [ms]')
    plt.ylabel('frequency [Hz]')
    ax4.tick_params(
        right = 'off',
        top = 'off'
    )


    plt.show()
    #"""
"""
    # Oppg 16: Squared Signal
    signal_square = time_signal**2
    fourier_square = np.fft.fft(signal_square) / len(signal_square)
    wavelet_diagram_squared, x_squared, y_squared = wavelet_analysis(fourier_square, frequencies, time)

    ax5 = plt.subplot(2, 1, 1)
    plt.title('Time Signal g(t)')
    ax5.plot(1000 * time, signal_square, c='darkblue')
    plt.xlabel('time [ms]')
    plt.ylabel('g(t)^2')
    ax5.tick_params(
        right = 'off',
        top = 'off'
    )


    ax6 = plt.subplot(2, 1, 2)
    plt.title('Fourier Spectrum of Squared Signal')
    ax6.plot(frequencies, np.real(fourier_square))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Fourier Coefficient')
    ax6.tick_params(
        right = 'off',
        top = 'off'
    )

    plt.subplots_adjust(hspace=0.9)

    plt.show()

    ax7 = plt.subplot(1, 1, 1, facecolor='white')
    plt.title('Wavelet Transform of the Squared Signal')
    plt.pcolormesh(1000 * x_squared, y_squared, np.abs(wavelet_diagram_squared), cmap='Blues')
    plt.colorbar()
    plt.xlabel('time [ms]')
    plt.ylabel('frequency [Hz]')
    ax7.tick_params(
        right = 'off',
        top = 'off'
    )

    plt.show()
"""
