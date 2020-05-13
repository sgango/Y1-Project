"""
Srayan Gangopadhyay
2020-05-13

Test implementation of Euler's method
"""

import scipy as sp

def func(x):  # RHS of ODE in form dy/dx= 
    return sp.exp(x)

# PARAMETERS
y0 = 1  # initial value
delta = 1  # step size
end = 10  # x-value to stop integration

steps = end/delta  # FIXME: may need to adjust this
time = sp.linspace(0, end, steps)

# TODO: write integrator