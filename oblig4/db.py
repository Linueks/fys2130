from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

threshold_for_hearing = 1e-12   #val 1e-12[w/m^2]
sound_intensity = np.linspace(threshold_for_hearing, 10**13 * threshold_for_hearing, 100000)    #val 1e-12 -> 10^13 * 1e-12
db = 10 * np.log10(sound_intensity / threshold_for_hearing)


plt.plot(sound_intensity, db)
plt.show()
