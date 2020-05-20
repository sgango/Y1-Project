"""
Using Runge-Kutta method to integrate Lorentz force equation
No elec. field, const. mag. field
Srayan Gangopadhyay
2020-05-20
"""

import numpy as np
import matplotlib.pyplot as plt

# PARAMETERS
x, y, z = 0, 0, 0  # initial position
vx, vy, vz = 1, 2, 3  # initial velocity
Bx, By, Bz = 0, 0, 1  # magnetic field
q, m = 1, 2  # charge, mass
h = 0.1  # step size
end = 6  # t-value to stop integration

def rk4(h, t, r, v):
    k1y = h * v
    k1v = h * func(x[i], y[i], v[i])
    k2y = h * (v[i] + 0.5*k1v)
    k2v = h * func((x[i] + 0.5*h), (y[i] + 0.5*k1v), (v[i] + 0.5*k1v))
    k3y = h * (v[i] + 0.5*k2v)
    k3v = h * func((x[i] + 0.5*h), (y[i] + 0.5*k2v), (v[i] + 0.5*k2v))
    k4y = h * (v[i] + k3v)
    k4v = h * func(x[i+1], (y[i] + k3v), (v[i] + k3v))
    
    y[i+1] = y[i] + (k1y + 2*k2y + 2*k3y + k4y) / 6
    v[i+1] = v[i] + (k1v + 2*k2v + 2*k3v + k4v) / 6

# EMPTY LISTS TO HOLD SOLUTION
x = []
y = []
z = []

for i in range(0, steps-1):
    # LORENTZ FORCE
    a_x = (q/m)*(vy*Bz - vz*By)
    b_y = (-q/m)*(vx*Bz - vz*Bx)
    a_z = (q/m)*(vx*By - vy*Bx)