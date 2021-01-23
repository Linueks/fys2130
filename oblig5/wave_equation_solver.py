from __future__ import division, print_function
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.style as style
import numpy as np
style.use('ggplot')



dx = 0.1
time_step = 0.1
wave_width=2.0
position = np.arange(-20, 20+dx, dx)
times = np.arange(0, 100, time_step)
wave_array = np.zeros((len(times), len(position)))

#factor = (velocity * dt/dx)**2
#initial_time_derivative = initial_wave * velocity * position / (2 * width**2)

def wave_form(x):
    initial_wave = np.exp(-(x**2 / (4 * wave_width**2)))

    return initial_wave


def wave_propagation(initial_wave):
    """
    Wave Equation in Difference form:
    u[t+1, i] = 2 * (1 - (v * dt/dx)**2) * u[t,:] - u[t,:]
                + (v * dt/dx)**2 * (u[t, i+1] + u[t, i-1]
    i is the index for the amplitude of the wave at all points and t is time
    """

    # Parameters and initial conditions
    velocity = 1.0
    initial_time_derivative = initial_wave * position * velocity / (2 * wave_width**2)
    wave_array[0, :] = initial_wave - time_step * initial_time_derivative
    wave_array[1, :] = initial_wave


    for t in range(1, len(times)-1):

        #"""
        for i in range(1, len(position)-1):
            wave_array[t+1, 1:-2] = 2 * (1 - (velocity * time_step / dx)**2) * wave_array[t, i] - wave_array[t-1, i]\
                            + (velocity * time_step / dx)**2 * (wave_array[t, i-1] + wave_array[t, i+1])
            #wave_array[t+1, 0] = 2 * (1 - (velocity * time_step / dx)**2) * wave_array[t, 0] - wave_array[t-1, 0]\
            #                + (velocity * time_step / dx)**2 * wave_array[t, 1]
            #wave_array[t+1,-1] = 2 * (1 - (velocity * time_step / dx)**2) * wave_array[t,-1] - wave_array[t-1,-1]\
            #                + (velocity * time_step / dx)**2 * wave_array[t,-2]
        #"""
        """
        wave_array[t+1, 2:len(position)-1] = 2 * (1 - (velocity * time_step / dx)**2) * wave_array[t, 2:len(position)-1] - wave_array[t-1, 2:len(position)-1]\
                        + (velocity * time_step / dx)**2 * (wave_array[t, 3:len(position)] + wave_array[t, 1:len(position)-2])
        # Boundary Conditions
        wave_array[t+1, 1] = 2 * (1 - (velocity * time_step / dx)**2) * wave_array[t, 1] - wave_array[t-1, 1] + (velocity * time_step / dx)**2 * wave_array[t, 2]
        wave_array[t+1, len(position)-1] = 2 * (1 - (velocity * time_step / dx)**2) * wave_array[t, len(position)-1] - wave_array[t-1, len(position)-1] + (velocity * time_step / dx)**2 * wave_array[t, len(position)-2]
        """


    fig = plt.figure()
    plts = []
    plt.xlim(-20, 20)
    plt.hold('on')
    plt.plot(position, wave_array[0,:], '--')
    plt.hold('off')
    for i in range(len(times)):
        p, = plt.plot(position, wave_array[i, :], 'k')
        plts.append([p])
    ani = animation.ArtistAnimation(fig, plts, interval=5, repeat_delay=3000)
    #ani.save('wave_prop_1.gif', writer='ImageMagick', fps=60)

    plt.show()

initial_wave = wave_form(position)
wave_propagation(initial_wave)
