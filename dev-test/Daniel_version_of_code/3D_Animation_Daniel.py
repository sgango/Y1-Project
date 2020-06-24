
"""

@author: Daniel 

"""
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #imports the required part to add the 3rd axis for plot to becomde 3D
import matplotlib.animation as animation
#####################################Initial Conditions
p0=[0,0,0]                                                #Initial position
v0=[1,1,1]                                                #Initial velocity
B=[0,0,1]                                                 #Initial magnetic field
E=[0,0,0]                                                 #Initial electric field
q=1                                                       #Charge
m=1                                                       #mass
dt=0.1                                                    #time interval
stop=100
steps=int(stop/dt)
###################################################
p=sp.zeros((steps,3))######Initial empty arrays
v=sp.zeros((steps,3))
v[0]=v0###Defining zeroth entry
p[0]=p0


##############################################################################

def cross(v1):                                            #Cross product function taking velocity and magnetic field
    return (q/m)*(E+sp.cross(v1,B))     


##############################################################################
for i in range(0,steps-1):                              #4th Order Runge-Kutta
    p1=dt*v[i]
    v1=dt*cross(v[i])
    p2=dt*(v[i]+0.5*v1)
    v2=dt*cross(v[i]+0.5*v1)    
    p3=dt*(v[i]+0.5*v2)
    v3=dt*cross(v[i]+0.5*v2)    
    p4 = dt * (v[i] + v3)
    v4 = dt * cross(v[i] + v3)    

    
    p[i+1]=p[i] + (p1 +2*p2 +2*p3 +p4)*(1/6)
    v[i+1]=v[i] + (v1 + 2*v2+2*v3+v4)*(1/6)
##############################################################################

fig = plt.figure()
FRAMES=100               #Number og grames

ax = Axes3D(fig)
##############################################################################
#ax.plot3D(p[:,0], p[:,1], p[:,2])      #Plots the stationary 3D curve
##############################################################################

def func(i):
    current_index = int(1000 / 10* i)
    ax.plot3D(p[:current_index, 0], p[:current_index, 1], p[:current_index, 2],color='g') #Function which is iterrated and creates the plot

    
ax.set_xlim3d([sp.amin(p[:,0]),sp.amax(p[:,0])]) #######Sets limits to the axis so that we can always see the whole plot
ax.set_xlabel('X')

ax.set_ylim3d([sp.amin(p[:,1]),sp.amax(p[:,1])]) 
ax.set_ylabel('Y')

ax.set_zlim3d([sp.amin(p[:,2]),sp.amax(p[:,2])]) 
ax.set_zlabel('Z')

ax.set_title('3D Test')
anim = animation.FuncAnimation(fig, func,
                               frames=FRAMES, interval=100) #This creates the animation

plt.show()
   
