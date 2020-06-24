"""
@author: Daniel Buguks
"""
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.axis as axi
from mpl_toolkits.mplot3d import Axes3D

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

p=sp.zeros((steps,3))######Initial empty arrays
v=sp.zeros((steps,3))######
v[0]=v0
p[0]=p0

##############################################################################

def cross(v1):                                            #Cross product function taking velocity and magnetic field
    return (q/m)*(E+sp.cross(v1,B))     


##############################################################################
# =============================================================================
# for k in range(0,steps-1):                              #Euler-Method, gets the'whirl-pool' plot
#     p1=dt*v[k]
#     v1=dt*cross(v[k])    
# 
# 
#     p[k+1]=p[k] + (p1)
#     v[k+1]=v[k] + (v1)
# plt.plot(p[:,0], p[:,1] ,label='Euler', color='m')
# =============================================================================
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

plt.xlabel("X Position")                                #Defining labeling variables
plt.ylabel("Y Position")
plt.title("2D - Runge Kutta")
plt.savefig('2D_Mag_Field.png')
plt.plot(p[:,0], p[:,1] ,label='Runge-Kutta', color='m')#plots the function
