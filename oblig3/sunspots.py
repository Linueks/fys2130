from __future__ import division, print_function
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
style.use('ggplot')


times = []
sunspots = []

# N = Tf_S = T/dt
# T = 318 years
# N = 318 then?


with open('solflekker.txt', 'r') as infile:
    for line in infile:
        cols = line.split()
        times.append(float(cols[0]))
        sunspots.append(float(cols[1]))

N = len(times)
X = np.fft.fft(sunspots) / N
freq_range = 0.5 * np.linspace(0, 1, N/2)



plt.subplot(211)
plt.plot(times, sunspots, '-b')
plt.ylabel('Number of Sun spots')
plt.xlabel('Time [years]')

plt.subplot(212)
plt.plot(freq_range, 2 * np.abs(np.real(X[0:round(N/2)])))
plt.ylabel("Fourier coefficient abs(X(f))")
plt.xlabel("Frequency [1/year]")
plt.ylim(-5, 50)

plt.subplots_adjust(hspace=0.5)
plt.show()
