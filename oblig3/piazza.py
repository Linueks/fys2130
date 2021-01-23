import numpy as np; import matplotlib.pyplot as plt;

def func(t, f1=1000.,f2=1600,c1=1.0,c2=1.7):
    return c1*np.sin(2*np.pi*f1*t) + c2*np.sin(2*np.pi*f2*t)

T=0.1; N=8192; dt=T/float(N)
t=np.linspace(0,T,N)

plt.plot(t,func(t))
plt.xlabel("Time")
plt.ylabel("f(t)")
plt.title("Original Signal")
plt.show()

#Fourier --------------------------
fs = 10000.
X = np.fft.fft(func(t))/float(N)
freq = np.fft.fftfreq(N,t[1]-t[0])

plt.plot(freq[:int(0.5*N)],np.abs(X[:int(0.5*N)]))
plt.title("Absolute Values of the Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("|X(freq)|")
plt.show()

#Wavelet -------------------------
fmin=800.; fmax=2000.; K=24.; n=1000
f_array = np.linspace(fmin,fmax,n)
omega_a_array = f_array*2*np.pi

def F_psi(omega_a, omega, K):
    one = 2*np.exp(-(K*(omega-omega_a)/float(omega_a))**2)
    two = - 2*np.exp(-K**2)*np.exp(-(K*omega/float(omega_a))**2)
    return one+two

W = np.zeros((n,int(N)),dtype=complex)
for i in range(n):
    W[i] = np.fft.ifft(F_psi(omega_a_array[i],2 * np.pi * freq,K)*X*N)

x,y = np.meshgrid(t,np.log10(f_array))
plt.pcolormesh(x,y,np.absolute(W))
plt.colorbar()
plt.show()
