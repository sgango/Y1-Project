"""
Test implementation of Euler's method
Srayan Gangopadhyay
2020-05-13
"""

import scipy as sp
import matplotlib.pyplot as plt

def func(y):  # RHS of ODE in form dy/dx= 
    return y

# PARAMETERS
y0 = 1  # initial value
delta = 1  # step size
end = 3  # x-value to stop integration

steps = int(end/delta)  # number of steps
x = sp.linspace(0, end, steps+1)  # array of x-values
y = sp.zeros(steps+1)  # empty array for solution
y[0] = y0  # inserting initial value

# INTEGRATING
for i in range(1, steps+1):
    y[i] = y[i-1] + (delta*func(y[i-1]))

plt.plot(x, y, label='Approx. soln (Euler)')
plt.plot(x, y, 'o')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()