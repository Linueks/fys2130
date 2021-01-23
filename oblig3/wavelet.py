'''
Fast fourier transform of a sound signal
Sebastian G. Winther-Larsen (2017)
'''

import numpy as np
from matplotlib import pyplot as plt
from progressbar import ProgressBar
from scipy import interpolate


pbar = ProgressBar()
#data, Fs = sf.read("gjok.wav")
#data, Fs = sf.read("svarttrost2.wav", start=17000)
#sd.play(data)


Fs = 1000
f1 = 1000
f2 = 1600
c1 = 1.0
c2 = 1.7
N = 8192

t = np.linspace(0, 10, N)
data = c1 * np.sin(2 * np.pi * f1 * t) + c2 * np.sin(2 * np.pi * f2 * t)

N = len(data)

mono_signal = data
fft_transform = np.fft.fft(mono_signal)
f = np.linspace(0, Fs*(N-1)/N, N)
nmax = round(N/2)

# Plotting signal
plt.plot(f[0:nmax], np.real(fft_transform[0:nmax]))
plt.show()

f_min = 800
f_max = 2000

K = 24

# Number of analyis frequencies
M = np.floor(np.log10(f_max / f_min) / np.log10(1 + (1 / (8*K)))) + 1
f_step = (f_max/f_min) ** (1 / (M-1))
f_analysis = f_min

T = N / Fs # Time of sound sample
t = np.linspace(0, T*(N-1)/N, N)

print("M = ", M, " N = ", N)
WLdiagram = np.zeros((int(M), int(N)))
f_used = np.zeros(int(M))

# Looping over all frequencies
print("Performing Wavelet Analysis")
for i in pbar(range(int(M))):
    factor = (K/f_analysis) * (K/f_analysis)

    # Creating wavelet
    FTwl = np.exp(-factor*(f-f_analysis) * (f-f_analysis))
    FTwl = FTwl - np.exp(-K*K) * np.exp(-factor*(f*f))
    FTwl = 2 * FTwl

    # Computing an entire row in wavelet diagram
    WLdiagram[i, :] = np.sqrt(np.abs(np.fft.ifft(FTwl * fft_transform)))

    f_used[i] = f_analysis # Storing saved frequencies

    f_analysis = f_analysis * f_step # Next frequency for analysis

# Reducing size of diagram in order to make plotting manageable
print("Reducing size of data for manageable plotting")
reduction_factor = 24
P = np.floor((K*Fs) / (reduction_factor * f_max))
NP = np.floor(N / P)
WLdiagram2 = np.zeros((int(M), int(NP)))
tP = np.zeros(int(NP))

for i in range(int(M)):
    for j in range(int(NP)):
        WLdiagram2[i, j] = WLdiagram[i, int(j*P)]
        tP[j] = t[int(j*P)]

plt.imshow(WLdiagram2, extent = [min(tP), max(tP),\
        np.log10(min(f_used)), np.log10(max(f_used))],
        aspect = 'auto', cmap="jet")
plt.show()
