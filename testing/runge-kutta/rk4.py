"""
Upgrading Euler method to 4th-order Runge Kutta
Srayan Gangopadhyay
2020-05-16
"""

import numpy as np
import matplotlib.pyplot as plt

"""
For a function of form y'' = f(x, y, y')
Define y' = v so y'' = v'
"""

def func(y, v, x):  # RHS of v' = in terms of y, v, x
    return x + v - 3*y

# PARAMETERS
y0 = 1  # y(x=0) = 
v0 = -2  # y'(x=0) = 
delta = 0.01  # step size
end = 4  # x-value to stop integration

steps = int(end/delta) + 1  # number of steps
x = np.linspace(0, end, steps)  # array of x-values (discrete time)
y = np.zeros(steps)  # empty array for solution
v = np.zeros(steps)
# TODO: initialise more empty arrays for other variables
y[0] = y0  # inserting initial value
v[0] = v0

# INTEGRATING
for i in range(1, steps):
    y[i] = y[i-1] + (delta*v[i-1])
    v[i] = v[i-1] + (delta*func(y[i-1], v[i-1], x[i-1]))
    # TODO: calculate and add other variables

plt.plot(x, y, label='Approx. soln (Euler)')
plt.plot(x, y, 'o')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
