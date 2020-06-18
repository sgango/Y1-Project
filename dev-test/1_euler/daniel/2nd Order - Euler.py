"""
Daniel Buguks
"""

### Defining function
def dxdt(x,v,t):
    return -6*v-9*x+sp.cos(3*t)

### Initial condition
t0=0
x0=0.5
v0=0

dt=0.1

stop=3
steps=int(stop/dx)    
t=sp.linspace(0,stop,steps)
xval=sp.zeros(steps)
vval=sp.zeros(steps)

xval[0]=x0
vval[0]=v0



### Loop for numerical solutions
for i in range(0,steps-1):
    xval[i+1]=xval[i] + dt*vval[i]
    vval[i+1]=vval[i] + dt*dxdt(xval[i],vval[i],t[i])


###Plotting our approximation
plt.plot(t, xval,'x' ,label='Euler', color='m')
plt.plot(t, xval, color='m')



###Plotting original solution
plt.plot(t,((0.5+(4*t)/3)*sp.exp(-3*t)+(1/18)*sp.sin(3*t)), label='original solutin', color='g')
plt.xlabel('t')
plt.ylabel('x')

plt.legend()
plt.show()
