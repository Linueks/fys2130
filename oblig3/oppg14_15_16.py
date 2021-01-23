from __future__ import division, print_function
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
style.use('ggplot')


N = 512
t = np.linspace(0, 1, N)
num_periods = 16




#x = np.sin(2 * np.pi * num_periods * t) # oppg 14, 15
x = np.sign(np.sin(2 * np.pi * num_periods * t)) # oppg 16 square wave
X = np.fft.fft(x, N) / N


freq_range = (N / 2) * np.linspace(0, 1, N/2)

plt.subplot(211)
plt.plot(t, x)
plt.ylabel("Signal Amplitude")
plt.xlabel('Time [s]')


plt.subplot(212)
plt.plot(freq_range, np.abs(np.real(X[0:round(N/2)])))
plt.ylabel("Fourier Coefficients, |X(f)|")
plt.xlabel('Frequency [Hz]')


plt.subplots_adjust(hspace=0.7)

plt.show()
