from __future__ import division, print_function
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
style.use('ggplot')

fs = 10000
f1 = 1000
f2 = 1600
c1 = 1.0
c2 = 1.7
N = 8192

t = np.linspace(0, 0.1, N)
x = c1 * np.sin(2 * np.pi * f1 * t) + c2 * np.sin(2 * np.pi * f2 * t)
X = np.fft.fft(x, N) / N


freq_range = (fs / 2) * np.linspace(0, 1, N/2)
print(np.abs(X[0]), np.mean(x))



plt.subplot(211)
plt.plot(t, x)
plt.ylabel("Signal Amplitude")
plt.xlabel('Time [s]')


plt.subplot(212)
plt.plot(freq_range[0:round(N/2)], 2 * np.abs(np.real(X[0:round(N/2)])))
plt.ylabel("Fourier Coefficients, |X(f)|")
plt.xlabel('Frequency [Hz]')


plt.subplots_adjust(hspace=0.7)

plt.show()
