"""
Using Runge-Kutta method to integrate Lorentz force equation
No elec. field, const. mag. field
Srayan Gangopadhyay
2020-05-20
"""

import numpy as np
import matplotlib.pyplot as plt

# PARAMETERS
r0 = [0, 0, 0]  # initial position
v0 = [1, 2, 3]  # initial velocity
B = [0, 0, 1]  # magnetic field
q, m = 1, 2  # charge, mass
h = 0.1  # step size
end = 6  # t-value to stop integration

def lorentz(vel):  # returns acceleration
    return (q/m)*np.cross(vel, B)

# RUNGE-KUTTA INTEGRATOR
def rk4(func, init1, init2, h, end):
    """
    Takes the RHS of a 2nd-order ODE with initial conditions,
     step size and end point, and integrates using the 4th-order
     Runge-Kutta algorithm. Returns solution in an array.

     r'' = f(t, r, v) where v = r'

     func: the function to be integrated
     init1: value of r at t=0
     init2: value of v at t=0
     h: step size
     end: t-value to stop integrating
    """

    steps = int(end/h)  # number of steps
    r = np.zeros((3, steps))  # empty matrix for solution
    v = np.zeros((3, steps))
    r[:,0] = init1  # inserting initial value
    v[:,0] = init2

    for i in range(0, steps-1):
        k1r = h * v[i]
        k1v = h * func(v[i])
        k2r = h * (v[i] + 0.5*k1v)
        k2v = h * func(v[i] + 0.5*k1v)
        k3r = h * (v[i] + 0.5*k2v)
        k3v = h * func(v[i] + 0.5*k2v)
        k4r = h * (v[i] + k3v)
        k4v = h * func(v[i] + k3v)
        r[i+1] = r[i] + (k1r + 2*k2r + 2*k3r + k4r) / 6
        v[i+1] = v[i] + (k1v + 2*k2v + 2*k3v + k4v) / 6
    return r

r = rk4(lorentz, r0, v0, h, end)