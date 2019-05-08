"""
Program: Dynamical Systems Template
File: dynamicalSystemsTemplate.pylab
Author: C.D. Wentworth
Version: 2-10-2018.1
Summary:
    Basic script for solving a system of first order differential equations
    that describes a dynamical system.
Usage: python dynamicalSystemsTemplate.py
Version History:
    1-15-2018.1: base
    12-11-2018.1: added some documentation
    
"""

import scipy.integrate as si
import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit

def Thrust(t):
    Tm=10.2
    Tb=2.5
    tm=0.27
    tb=0.45
    tf=1.5
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

#   Unpack the parameter list
    
    # mass of rocket (kg)
    m = 0.020
    # gravity magnitude (m/s^2)
    g = 9.81
    # density of air (kg/m^3)
    rho = 1.225
    # drag coefficient
    Cd = 0.75
    # cross sectional area of rocket (m^2)
    A = (4.6e-4)
    # Thrust force in terms of time (N)
    T = Thrust(t)
    # Drag force in terms of time (N)
    D = 0.5*(rho*Cd*A*(y1**2))*np.sign(y1)

#   Calculate the rates of change (the derivatives)
    dy0dt = y1
    dy1dt = (1/m)*(T-D-(m*g))

    return [dy0dt,dy1dt]

#--Main Program

# Define the initial conditions
y_0 = [0.0,0.0]

# Define the time grid
t = np.linspace(0,20,100)

# Solve the DE
p = ()
sol = si.odeint(calc_RHS,y_0,t,args=(p,))
y0 = sol[:,0]
y1 = sol[:,1]

# Plot the solution
plt.plot(t,y0)
plt.xlabel('t (s)',fontsize=14)
plt.ylabel('y (m)',fontsize=14)
plt.title('Model Rocket Run 1',fontsize=20,color='Brown')
plt.xlim(0,20)
plt.ylim(0,275)
plt.savefig('ModelRocketData1.png')
plt.show()