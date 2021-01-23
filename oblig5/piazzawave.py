import numpy as np
import matplotlib.pyplot as plt

class Wave(object):
    def __init__(self, x0 = -20, x1 = 20, dx = 0.1, sigma = 2, v = 0.5, dt = 0.1):
        N = (abs(x0) + abs(x1))/dx + 1
        self.x = np.linspace(x0, x1, N)
        self.dx = dx
        self.sigma = sigma
        self.v = v
        self.dt = dt

    def draw_wave(self):
        u = np.exp(-(self.x/(2*self.sigma))**2)
        dudt = (self.v/(2*self.sigma**2))*self.x*u
        u0 = u - dudt*self.dt
        un = np.zeros(len(u))
        fact = (self.dt*self.v/self.dx)**2
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        line, = ax.plot(self.x, u)
        line2, = ax.plot(self.x, u)
        plt.draw()

        for i in range(500):
            un[1:-2] = (2*(1 - fact))*u[1:-2] - u0[1:-2] + fact*(u[2:-1] + u[0:-3])
            un[0] = (2*(1 - fact))*u[0] - u0[0] + fact*u[1]
            un[-1] = (2*(1 - fact))*u[-1] - u0[-1] + fact*u[-2]
            line.set_ydata(un)
            plt.draw()
            plt.pause(0.001)
            u0 = u
            u = un
        plt.ioff()
        plt.show()
if __name__ == '__main__':
    a = Wave()
    a.draw_wave()
