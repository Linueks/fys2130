from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import scipy.constants as const
style.use('ggplot')

k = 100
m = 10
omega = np.sqrt(k/m)
B = 3
C = 5

def z(t):
    return B * np.sin(omega * t) + C * np.cos(omega * t)

def v(t):
    return B * omega * np.cos(omega * t) - C * omega * np.sin(omega * t)


t = np.linspace(0, 10, 1000)

plt.xlabel('Posisjon z [m]')
plt.ylabel('Hastighet v [m/s]')
plt.axis('equal')
plt.plot(v(t), z(t))
plt.savefig('vel_vs_pos.png')
