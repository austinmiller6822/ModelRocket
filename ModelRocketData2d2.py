# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 13:40:53 2019

@author: Austin
"""

import scipy.integrate as si
import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit

def Thrust(t):
    Tm=6.2
    Tb=2.0
    tm=0.15
    tb=0.33
    tf=0.55
    if t < tm:
        T=((Tm/tm)*(t))
    elif t < tb:
        s=(Tb-Tm)/(tb-tm)
        T=(s*t)+(Tm-s*tm)
    elif t < tf:
        T=Tb
    else:
        T=0
    return T

# Create a function that defines the rhs of the differential equation system
def calc_RHS(y,t,p):
#   y = a list that contains the system state
#   t = the time for which the right-hand-side of the system equations
#       is to be calculated.
#   p = a tuple that contains any parameters needed for the model
#
    import numpy as np

#   Unpack the state of the system
    y0 = y[0]
    y1 = y[1]
    y2 = y[2]
    y3 = y[3]

#   Unpack the parameter list
    m,k,theta_0 = p
    # mass of rocket (kg)
    m = 0.020
    # gravity magnitude (m/s^2)
    g = 9.81
    # velocity (m/s)
    v= ((y1**2)+(y3**2))**1/2
    # density of air (kg/m^3)
    rho = 1.225
    # drag coefficient
    Cd = 0.75
    # cross sectional area of rocket (m^2)
    A = (4.6e-4)
    # Thrust force in terms of time (N)
    T = Thrust(t)
    # Drag force in terms of time (N)
    D = (0.5*(rho*Cd*A*(v**2)))
    # Length (m)
    L = 0.75
    Lx = L*np.sin(theta_0)
    Ly = L*np.cos(theta_0)
    
# Calculate the rates of change (the derivatives)   
    if y2<Lx:
        sin0=np.sin(theta_0)
        cos0=np.cos(theta_0)
        dy0dt = y1
        dy1dt = (1/m)*(T-D*cos0-(m*g)*sin0)*cos0
        dy2dt = y3
        dy3dt = (1/m)*(T-D*sin0-(m*g)*cos0)*sin0
    else:
        cosTheta = np.abs(y1/v)
        sinTheta = np.abs(y3/v)
        dy0dt = y1
        dy1dt = (1/m)*(T-D*cosTheta*np.sign(y1)-m*g)
        dy2dt = y3
        dy3dt = (1/m)*(T-D*sinTheta) 
    return [dy0dt,dy1dt,dy2dt,dy3dt]

#--Main Program

# Define the initial conditions
y_0 = [0.0,0.0,0.0,0.0]

# Define the time grid
t = np.linspace(0,10,300)

# Define model parameters
m = 0.020
k = 2.0e-3
theta_0 = 0.30
p = m,k,theta_0

# Solve the DE
sol = si.odeint(calc_RHS,y_0,t,args=(p,))
y0 = sol[:,0]
y1 = sol[:,1]
y2 = sol[:,2]
y3 = sol[:,3]

# Plot the solution
p1=plt.subplot(2,1,1)
p1.plot(t,y0,color='g')
p1.set_ylim({0,20})
p1.set_xlim({0,4})
p1.set_ylabel('y (m)')
p2=plt.subplot(2,1,2)
p2.plot(t,y2,color='b')
p2.set_ylim({0,40})
p2.set_xlim({0,4})
p2.set_ylabel('x (m)')
p2.set_xlabel('t (s)')
plt.savefig('ModelRocketData2D2.png')
plt.show()