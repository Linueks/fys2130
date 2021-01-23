from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from matplotlib.pyplot import style
style.use('ggplot')

# time shit
dt = 0.001
total_time = 10                                                  # [s]
total_time_steps = int(total_time / dt)


# physical quantities
mass = 0.1                                                      # [kg]
spring_coeff = 10                                               # [N/m]
drag_coeff = 0.1                                                # [kg/s]
g = const.g                                                     # [m/s^2]
equillibrium_length = -mass * g / spring_coeff                  # [m]
init_y = 0.1                                                    # [m]



# coordinate system
unit_x = np.array([1, 0])
unit_y = np.array([0, 1])


# arrays for integration
position = np.zeros((2, total_time_steps))
velocity = np.zeros((2, total_time_steps))
acceleration = np.zeros((2, total_time_steps))


# initials
position[:, 0] = np.array([0, init_y])
velocity[:, 0] = np.array([0, 0])                   # dosen't need to be explicitly stated but w/e symmetry


def spring_pendulum(pos, vel, t):
    """
    Gets the acceleration of the mechanical spring pendulum from the sum of the
    forces acting on the pendulum at the current time step. I originally thought the
    task was to create an elastic pendulum, so there are some remants from that in the code.
    """
    unit_r = pos / np.linalg.norm(pos)

    F_gravity = -mass * g * unit_y                                              # always points downwards
    F_spring = -spring_coeff * (np.linalg.norm(pos)) * unit_r                   # always points radially
    F_drag = -drag_coeff * vel                                                  # always points in negative v direction

    acceleration[:, t] = (F_gravity + F_spring + F_drag) / mass


    return acceleration[:, t]


def diffEq(current_pos, current_vel, current_time_step):
    """
    Takes the value for the position and velocity at the current time step
    and returns the the value for the acceleration.
    """
    current_acceleration = spring_pendulum(current_pos, current_vel, current_time_step)


    return current_acceleration


def RK4(init_pos, init_vel, current_time_step):
    """
    Method is meant to solve for the motion of a mechanical spring pendulum.
    """
    # First sample of acceleration
    acceleration_1 = diffEq(init_pos, init_vel, current_time_step)   # Seems like i need time dependence, but accel is not dependent on time?
    velocity_1 = init_vel

    # First half step
    position_half_1 = init_pos + velocity_1 * dt / 2
    velocity_half_1 = init_vel + acceleration_1 * dt / 2

    # Second sample of acceleration
    acceleration_2 = diffEq(position_half_1, velocity_half_1, current_time_step)   # Seems like i need time dependence, but accel is not dependent on time?
    velocity_2 = velocity_half_1

    # Second half step
    position_half_2 = init_pos + velocity_2 * dt / 2
    velocity_half_2 = init_vel + acceleration_2 * dt / 2

    # Third sample of acceleration
    acceleration_3 = diffEq(position_half_2, velocity_half_2, current_time_step)   # Seems like i need time dependence, but accel is not dependent on time?
    velocity_3 = velocity_half_2

    # Final step, this is not really a half step, but symmetric structure to the other half steps
    position_half_3 = init_pos + velocity_3 * dt
    velocity_half_3 = init_vel + acceleration_3 * dt

        # Fourth sample of acceleration
    acceleration_4 = diffEq(position_half_3, velocity_half_3, current_time_step)   # Seems like i need time dependence, but accel is not dependent on time?
    velocity_4 = velocity_half_3

        # Getting the middle values
    acceleration_middle = 1 / 6 * (acceleration_1 + 2*acceleration_2 + 2*acceleration_3 + acceleration_4)
    velocity_middle = 1 / 6 * (velocity_1 + 2*velocity_2 + 2*velocity_3 + velocity_4)

    next_position = init_pos + velocity_middle * dt
    next_velocity = init_vel + acceleration_middle * dt


    return next_position, next_velocity


def integrator_loop(plot=True):
    """
    Function should loop over total time steps and for each one calculate the acceleration using the
    spring_pendulum method. Once accel. is obtained the RK4 method integrates to find the position and velocity
    one time step after and pushes 'it' along...
    """


    for t in xrange(total_time_steps-1):
        position[:, t+1] = RK4(position[:, t], velocity[:, t], t)[0]
        velocity[:, t+1] = RK4(position[:, t], velocity[:, t], t)[1]

    if plot:
        time = np.linspace(0, total_time, total_time_steps)
        #plt.axis('equal')
        #plt.plot(position[0, :], position[1, :])
        plt.plot(time, position[1, :])
        plt.axhline(equillibrium_length, xmin=0, xmax=1)

        plt.xlabel('Time [s]')
        plt.ylabel('Height [m]')
        plt.legend(['Height vs. Time, h=0=L0', 'Equillibrium Height L1'])
        plt.title('Mechanical Spring Pendulum')
        plt.show()


integrator_loop()
