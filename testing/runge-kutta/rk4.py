"""
Upgrading Euler method to 4th-order Runge-Kutta
for 2nd-order ODEs
Srayan Gangopadhyay
2020-05-16
"""

import numpy as np
import matplotlib.pyplot as plt

"""
y' = dy/dx
For a function of form y'' = f(x, y, y')
Define y' = v so y'' = v'
"""

def func(x, y, v):  # RHS of v' = in terms of x, y, v
    return x + v - 3*y

# PARAMETERS
y0 = 1  # y(x=0) = 
v0 = -2  # y'(x=0) = 
h = 0.01  # step size
end = 4  # x-value to stop integration

steps = int(end/h)  # number of steps
x = np.linspace(0, end, steps)  # array of x-values (discrete time)
y = np.zeros(steps)  # empty array for solution
v = np.zeros(steps)

y[0] = y0  # inserting initial value
v[0] = v0

# INTEGRATING
# https://mathworld.wolfram.com/Runge-KuttaMethod.html
for i in range(0, steps-1):
    k1v = h * func(x[i], y[i], v[i])
    k2v = h * func((x[i] + 0.5*h), (y[i] + 0.5*k1v), (v[i] + 0.5*k1v))
    k3v = h * func((x[i] + 0.5*h), (y[i] + 0.5*k2v), (v[i] + 0.5*k2v))
    k4v = h * func(x[i+1], (y[i] + k3v), (v[i] + k3v))

    # TODO: calculate ks for y and add below
    
    v[i+1] = v[i] + (h*func(x[i], y[i], v[i]))
    y[i+1] = y[i] + (h*v[i])

plt.plot(x, y, label='Approx. soln (Euler)')
plt.plot(x, y, 'o')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
