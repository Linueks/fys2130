from __future__ import division
import numpy as np

L = 25e-6
R = 1
C = 100e-9
Q = 15.8


top = L * (1/(L*C) - ((1/np.sqrt(L*C)) + (1 / (np.sqrt(L*C) * 2*Q)))**2)
bot = R * ((1/np.sqrt(L*C)) + (1 / (np.sqrt(L*C) * 2*Q)))

print top / bot
