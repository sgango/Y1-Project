"""
y1project.py
Srayan Gangopadhyay
2020-06-17
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML, display  # show anim. in ntbk
from tabulate import tabulate  # pretty text output
from datetime import datetime  # timestamp for output files
from tqdm.auto import tqdm  # progress bar

# CONSTANTS
m_pr = 1.67e-27  # proton mass (kg)
m_el = 9.109e-31  # electron mass (kg)
e = 1.602e-19  # elementary charge (C)
Re = 6.37e6  # Earth radius (m)

def lorentz(r, vel, E, B, q, m):
    """The Lorentz force equation. Returns
    acceleration of particle.

    Parameters:
    r - 3D position vector
    vel - 3D velocity vector
    E - electric field function
    B - magnetic field function
    q - charge
    m - mass

    All in standard SI units."""
    return (q/m)*(E(r) + np.cross(vel, B(r)))

def rk4(func, init1, init2, h, end, E_field, B_field, q, m, size):
    """Takes the RHS of a 2nd-order ODE with initial conditions,
    step size and end point, and integrates using the 4th-order
    Runge-Kutta algorithm. Returns solution in an array.

    r'' = f(t, r, v) where v = r'

    func: the function to be integrated
    init1: value of r at t=0
    init2: value of v at t=0
    h: step size
    end: t-value to stop integrating
    E_field: electric field function
    B_field: magnetic field function
    q: charge
    m: mass
    size: simulation dimensions (array/list)
    """
    steps = int(end/h)  # number of steps
    r = np.zeros((3, steps))  # empty matrix for solution
    v = np.zeros((3, steps))
    r[:,0] = init1  # inserting initial value
    v[:,0] = init2

    for i in tqdm(range(0, steps-1), desc='Integrating'):
        k1r = h * v[:,i]
        k1v = h * func(r[:,i], v[:,i], E_field, B_field, q, m)
        k2r = h * (v[:,i] + 0.5*k1v)
        k2v = h * func(r[:,i], v[:,i] + 0.5*k1v, E_field, B_field, q, m)
        k3r = h * (v[:,i] + 0.5*k2v)
        k3v = h * func(r[:,i], v[:,i] + 0.5*k2v, E_field, B_field, q, m)
        k4r = h * (v[:,i] + k3v)
        k4v = h * func(r[:,i], v[:,i] + k3v, E_field, B_field, q, m)
        new_r = r[:,i] + (k1r + 2*k2r + 2*k3r + k4r) / 6
        new_v = v[:,i] + (k1v + 2*k2v + 2*k3v + k4v) / 6

        if (new_r[0] < size[0]*-0.5):  # stop particle leaving box
            new_r[0] += size[0]
        if (new_r[0] >= size[0]*0.5):
            new_r[0] -= size[0]
        if (new_r[1] < size[1]*-0.5):
            new_r[1] += size[1]
        if (new_r[1] >= size[1]*0.5):
            new_r[1] -= size[1]
        if (new_r[2] < size[2]*-0.5):
            new_r[2] += size[2]
        if (new_r[2] >= size[2]*0.5):
            new_r[2] -= size[2]

        r[:,i+1] = new_r
        v[:,i+1] = new_v
    return r, v

def calcmod(r, v):
    """Calculates modulus of vectors.
    Returns arrays of distances and speeds."""
    distances = np.linalg.norm(r, axis=0)
    speeds = np.linalg.norm(v, axis=0)
    return distances, speeds

def printout(r, v):
    """Prints out basic stats in table
    form."""
    distances, speeds = calcmod(r, v)
    print("\n",tabulate([['Max. distance', np.amax(distances)],
                     ['Avg. distance', np.mean(distances)],
                     ['Max. speed', np.amax(speeds)],
                     ['Avg. speed', np.mean(speeds)]],
                    headers=['Parameter', 'Value']))

def textout(r, v):
    """Exports raw data to .csv file."""
    distances, speeds = calcmod(r, v)
    filename = (datetime.now()
                .strftime("%Y%m%d-%H%M%S") + "_data.csv")
    with open(filename, 'a') as f:
        f.write("r-vectors\n")
        np.savetxt(f, r, delimiter=",")
        f.write("\nv-vectors\n")
        np.savetxt(f, v, delimiter=",")
        f.write("\ndistances\n")
        np.savetxt(f, distances[None], delimiter=",")
        f.write("\nspeeds\n")
        np.savetxt(f, speeds[None], delimiter=",")
    print("\nExported full data to ",filename)

def plotsetup(r):
    """Returns figure and set of 3D axes
    ready for plotting/animation. Axis limits
    are automatically set, given an array
    of position vectors.
    """
    fig = plt.figure()  # generate a figure
    ax = fig.add_subplot(111, projection='3d')  # set up 3d axes
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(30, -113)  # change viewing angle
    ax.set_xlim3d(np.amin(r[0]), np.amax(r[0]))  # auto-scale axes
    ax.set_ylim3d(np.amin(r[1]), np.amax(r[1]))
    ax.set_zlim3d(np.amin(r[2]), np.amax(r[2]))
    return (fig, ax)

def animate(i, ax, r, pbar):
    """Called by FuncAnimation to plot each frame."""
    j = 100*i  # to skip frames and change animation speed
    ax.plot3D(r[0, :j], r[1, :j], r[2, :j], ',', color='magenta')
    pbar.update(1)  # increment progress bar

def plot_or_anim(r, frames, fig, ax, anim):
    if not anim:
        ax.plot(r[0], r[1], r[2], ',', color='magenta')
        plt.show()
    else:
        pbar = tqdm(total=frames+1, desc='Animating')  # start progress bar
        animat = animation.FuncAnimation(fig, animate, fargs=(ax, r, pbar),
                                         frames=frames, interval=50,
                                         blit=False, repeat=False)
        display(HTML(animat.to_html5_video()))
        pbar.close()  # close progress bar
