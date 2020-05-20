"""
Using Runge-Kutta method to integrate Lorentz force equation
No elec. field, const. mag. field
Srayan Gangopadhyay
2020-05-20
"""

import numpy as np
import matplotlib.pyplot as plt

def x-lorentz(t, x, vx):  # x-component of equation of motion
    return (q/m)*(vy*Bz - vz*By)
    # TODO: add electric field

def y-lorentz(t, y, vy):
    return (-q/m)*(vx*Bz - vz*Bx)

def z-lorentz(t, z, vz):
    return (q/m)*(vx*By - vy*Bx)

# TODO: put this integrator in a separate module
def rk4(func, init1, init2, h, end):
    # FIXME: change variable names in integrator
    # (do we need to get function arguments?)
    """
    Takes the RHS of a 2nd-order ODE with initial conditions,
     step size and end point, and integrates using the 4th-order
     Runge-Kutta algorithm. Returns solution in an array.
    """
    steps = int(end/h)  # number of steps
    x = np.linspace(0, end, steps)  # array of x-values (discrete time)
    y = np.zeros(steps)  # empty array for solution
    v = np.zeros(steps)
    y[0] = y0  # inserting initial value
    v[0] = v0
    for i in range(0, steps-1):
        k1y = h * v[i]
        k1v = h * func(x[i], y[i], v[i])
        k2y = h * (v[i] + 0.5*k1v)
        k2v = h * func((x[i] + 0.5*h), (y[i] + 0.5*k1v), (v[i] + 0.5*k1v))
        k3y = h * (v[i] + 0.5*k2v)
        k3v = h * func((x[i] + 0.5*h), (y[i] + 0.5*k2v), (v[i] + 0.5*k2v))
        k4y = h * (v[i] + k3v)
        k4v = h * func(x[i+1], (y[i] + k3v), (v[i] + k3v))
        y[i+1] = y[i] + (k1y + 2*k2y + 2*k3y + k4y) / 6
        v[i+1] = v[i] + (k1v + 2*k2v + 2*k3v + k4v) / 6
    return y

# PARAMETERS
# Physics
x0, y0, z0 = 0, 0, 0  # initial position of particle
vx0, vy0, vz0 = 1, 1, 1  # initial velocity
Bx, By, Bz = 0, 0, 1  # magnetic field
q, m = 1, 2  # charge, mass
# Runge-Kutta
h = 0.1  # step size
end = 6  # t-value to stop integration
steps = int(end/h)  # number of steps
t = np.linspace(0, end, steps)  # array of t-values (discrete time)