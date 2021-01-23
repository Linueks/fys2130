from __future__ import division, print_function
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
style.use('ggplot')

Fs = 1000
dt = 1 / Fs
N = 1024
t = np.arange(N) * dt
freq = 2200

x = 0.8 * np.cos(2 * np.pi * freq * t)

plt.subplot(211)
plt.title("FFT, %i Hz Signal" % freq)
plt.plot(Fs * t, x)
plt.ylabel("Original signal, Hz")
plt.xlabel('Time [s]')


X = np.fft.fft(x, N) / N
freq_range = (Fs/2) * np.linspace(0, 1, N/2)


plt.subplot(212)
plt.plot(freq_range, 2 * np.abs(np.real(X[0:round(N/2)])))
plt.ylabel("Fourier spectrum, amplitude")
plt.xlabel("Frequency [Hz]")
plt.savefig('ff%i.jpg' % freq)
