
"""

@author: Daniel 

"""
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
#####################################Initial Conditions
p0=[0,7,0]                                                #Initial position
v0=[1,1,1]                                                #Initial velocity
B=[0,0,1]                                                 #Initial magnetic field
E=[0,0,0]                                                 #Initial electric field
q=1                                                       #Charge
m=1                                                       #mass
dt=0.1                                                    #time interval
stop=100
cube=[10,10,10]
steps=int(stop/dt)
###################################################

p=sp.zeros((steps,3))######Initial empty arrays
v=sp.zeros((steps,3))
v[0]=v0###Defining zeroth entry
p[0]=p0          #time interval#

##############################################################################

def B_field(p,q,v):     #Defining magnetic field function with position, charge, velocity
    B=q*sp.cross(v,p)/np.linalg.norm(p[1])
    return B

##############################################################################

def cross(v1):
    return (q/m)*(E+sp.cross(v1,B))      #Cross product function

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
    
    if (p[i,0] <  -cube[0] * 0.5):                                                                                          ###METHOD 1
        p[i,0] = p[i,0] +   cube[0]
    if (p[i,0] >=  cube[0] * 0.5): 
        p[i,0] = p[i,0] - cube[0]
    if (p[i,1] <  -cube[1] * 0.5): 
        p[i,1] = p[i,1] +   cube[1]
    if (p[i,1] >=  cube[1] * 0.5): 
        p[i,1] = p[i,1] - cube[1]
    if (p[i,2] <  -cube[2] * 0.5): 
        p[i,2] = p[i,2] +   cube[2]
    if (p[i,2] >=  cube[2] * 0.5): 
        p[i,2] = p[i,2] - cube[2]
        
 #############################################################################       

#     p[i,0] = p[i,0] % cube[0]
#     p[i,1] = p[i,1] % cube[1]################METHOD 2
#     p[i,2] = p[i,2] % cube[2]
        
############################################################################# Masked arrays, REMOVES DISCONTINUITIES JUMPS

p[:,0]=np.ma.masked_where((cube[0]>p[:,0])&(-cube[0]<p[:,0]),p[:,0])
p[:,1]=np.ma.masked_where((cube[1]>p[:,1])&(-cube[1]<p[:,1]),p[:,1])
p[:,2]=np.ma.masked_where((cube[2]>p[:,2])&(-cube[2]<p[:,2]),p[:,2])
disc11 = np.where(np.abs(np.diff(p[:,0])) >= cube[0]-1)[0]
disc22= np.where(np.abs(np.diff(p[:,1])) >= cube[1]-1)[0]
disc33 = np.where(np.abs(np.diff(p[:,2])) >= cube[2]-1)[0]
disc1=np.array(disc11)
disc2=np.array(disc22)
disc3=np.array(disc33)
p[disc1,0]=np.nan
p[disc2,0]=np.nan
p[disc3,0]=np.nan
p[steps-1,0]=np.nan
p[steps-1,1]=np.nan
p[steps-1,2]=np.nan

###################################################################################

fig = plt.figure()
ax = Axes3D(fig)
#ax.plot3D(p[:,0], p[:,1], p[:,2])  #Plots the 3D graph - Stationary
def func(k):  #########One of the input parameters needed for the animation.FuncAnimated which is used for iterating over the new points
    step=10*k #The step size regulates the speed of the animation, very easy to change, just increase integer to speed up and decrease to slow down
    ax.plot3D(p[:step, 0], p[:step, 1], p[:step, 2],color='g')#For each i integrates the new function
###################################################################################
    
###Degfining the axes    
ax.set_xlim3d([sp.nanmin(p[:,0]),sp.nanmax(p[:,0])]) ########ABLE TO PLOT AXIS EXCLUDING NAN VALUES
ax.set_xlabel('X')

ax.set_ylim3d([sp.nanmin(p[:,1]),sp.nanmax(p[:,1])])
ax.set_ylabel('Y')

ax.set_zlim3d([sp.nanmin(p[:,2]),sp.nanmax(p[:,2])])
ax.set_zlabel('Z')

###################################################################################
# ax.set_xlim3d([sp.amin(p[:,0]),sp.amax(p[:,0])]) #Axes without considering the nan values
# ax.set_xlabel('X')
# 
# ax.set_ylim3d([sp.amin(p[:,1]),sp.amax(p[:,1])])
# ax.set_ylabel('Y')
# 
# ax.set_zlim3d([sp.amin(p[:,2]),sp.amax(p[:,2])])
# ax.set_zlabel('Z')
###################################################################################
lorentz_3D = animation.FuncAnimation(fig, func, frames=100, interval=20,blit=False, repeat=False,save_count=30) ###Actual function which takes in the arguments and plots the animation
plt.show()
