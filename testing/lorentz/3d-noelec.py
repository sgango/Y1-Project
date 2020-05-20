"""
Using Runge-Kutta method to integrate Lorentz force equation
No elec. field, const. mag. field
Srayan Gangopadhyay
2020-05-20
"""

import numpy as np
import matplotlib.pyplot as plt

# PARAMETERS
r0 = [0, 0, 0]  # initial position of particle: x, y, z-components
v0 = [0, 0, 0]  # initial velocity
B = [0, 0, 1]  # magnetic field
q, m = 1, 2  # charge, mass

h = 0.1  # step size
end = 6  # t-value to stop integration

# LORENTZ FORCE EQUATIONS
def x_lorentz(t, v, B):  # x-component of equation of motion
    return (q/m)*(v[1]*B[2] - v[2]*B[1])

def y_lorentz(t, v, B):
    return (-q/m)*(v[0]*B[2] - v[2]*B[0])

def z_lorentz(t, v, B):
    return (q/m)*(v[0]*B[1] - v[1]*B[0])

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
    t = np.linspace(0, end, steps)  # array of x-values (discrete time)
    r = np.zeros(steps)  # empty array for solution
    v = np.zeros(steps)
    r[0] = init1  # inserting initial value
    v[0] = init2

    for i in range(0, steps-1):
        k1r = h * v[i]
        k1v = h * func(t[i], r[i], v[i])
        k2r = h * (v[i] + 0.5*k1v)
        k2v = h * func((t[i] + 0.5*h), (r[i] + 0.5*k1v), (v[i] + 0.5*k1v))
        k3r = h * (v[i] + 0.5*k2v)
        k3v = h * func((t[i] + 0.5*h), (r[i] + 0.5*k2v), (v[i] + 0.5*k2v))
        k4r = h * (v[i] + k3v)
        k4v = h * func(t[i+1], (r[i] + k3v), (v[i] + k3v))
        r[i+1] = r[i] + (k1r + 2*k2r + 2*k3r + k4r) / 6
        v[i+1] = v[i] + (k1v + 2*k2v + 2*k3v + k4v) / 6
    return r

# PASS LORENTZ EQUATION INTO INTEGRATOR
x = rk4(x_lorentz, r0[0], v0[0], h, end)
y = rk4(y_lorentz, r0[1], v0[2], h, end)
z = rk4(z_lorentz, r0[1], v0[2], h, end)

plt.plot(x, y, label='Approx. soln (RK4)')
plt.plot(x, y, 'o')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
