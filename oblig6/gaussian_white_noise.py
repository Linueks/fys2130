'''
Program that generates Gaussian white noise.
Each frequency component is generated randomly b/w 0  and
the a value assigne by Normal distribution.
Each phase component is generated randombly b/w 0 and 2pi.
The finished signal is Fourier transformed.
Sebastian G. Winther-Larsen (2017)
'''
import numpy as np
from matplotlib import pyplot as plt

def gaussian_white_noise(f_sample, N, f_center, f_width):

    # Parameters and necessary arrays
    f_sigma = f_width / 2
    y = np.zeros(N, 'complex')
    T = N / f_sample
    t = np.linspace(0, T*(N-1)/N, N)
    f = np.linspace(0, f_sample*(N-1)/N, N)
    n_center = np.floor(N*f_center / (f_sample*(N-1) / N))
    n_sigma = np.floor(N*f_sigma / (f_sample*(N-1) / N))

    # Computations
    gauss = np.exp(-(f - f_center) * (f - f_center) / (f_sigma*f_sigma))
    amplitude = np.random.rand(N) * gauss
    phase = np.random.rand(N) * 2*np.pi
    y.real = amplitude * np.cos(phase)
    y.imag = amplitude * np.sin(phase)

    # Must mirror lower half to get correct result
    n_half = np.round(N/2)

    for i in range(int(n_half - 1)):
        #if (i == 0 or i == (n_half - 2)):
            #print("First index: ", N - i - 1)
            #print("Second index: ", i + 1)

        y[int(N - i - 1)] = np.conjugate(y[int(i + 1)])

    y[int(n_half)] = y[int(n_half)].real
    y[0] = 0.0

    q = np.real(np.fft.ifft(y)*200)

    return y, q

if __name__ == '__main__':
    y, q = gaussian_white_noise(44100, 2**16, 5000, 500)
    #plt.plot(y.real)
    plt.plot(q)
    plt.show()
