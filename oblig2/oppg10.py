from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import scipy.constants as const
style.use('ggplot')


m = 1
g = -const.g
init_height = 10
total_time = 10
dt = 0.000001
total_time_steps = int(total_time / dt)

velocity = [0]
position = [init_height]
time = np.linspace(0, total_time, total_time_steps+1)


for t in xrange(total_time_steps):
    if position[t] <= 0:
        velocity.append(-(velocity[t] + g * dt))
    else:
        velocity.append(velocity[t] + g * dt)

    position.append(position[t] + velocity[t+1] * dt)




plt.xlabel('Posisjon z [m]')
plt.ylabel('Hastighet v [m/s]')
plt.axis('equal')
plt.plot(position, velocity)
plt.savefig('ball_vel_vs_pos.png')
