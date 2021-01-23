from __future__ import division, print_function
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
style.use('ggplot')

sample_freq = 1000
dt = 1 / sample_freq
number_of_samples = 1024

time = np.arange(number_of_samples) * dt
frequency = 0.5 * sample_freq * np.linspace(0, 1, 0.5 * number_of_samples)


# generating signal and transforming
signal = 0.7 * np.sin(2 * np.pi * 400 * time) + np.cos(2 * np.pi * 120 * time)

transformed_signal = np.fft.fft(signal, number_of_samples) / number_of_samples

half_spectrum = np.real(transformed_signal[0:round(0.5 * number_of_samples)])

print(half_spectrum[0])
print(np.mean(signal))



plt.subplot(211)
plt.title("FFT")
plt.plot(time, signal)
plt.ylabel("Original signal, Hz")
plt.xlabel("t [s]")

plt.subplot(212)
plt.plot(frequency, np.abs(half_spectrum))
plt.ylabel("Fourier spectrum, amplitude")
plt.xlabel("Frequency [Hz]")
plt.show()


#signal = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
#print(' '.join('%5.3f' % np.abs(f) for f in np.fft.fft(signal)))
