"""
Srayan Gangopadhyay
2020-05-13

Test implementation of Euler's method
"""

import scipy as sp

def func(x):  # RHS of ODE in form dy/dx= 
    return x**2

# PARAMETERS
x0 = 0  # initial value
delta = 0.01  # step size
end = 10  # x-value to stop integration

# TODO: discrete time array