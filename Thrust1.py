# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:18:49 2019

@author: Austin
"""
import numpy as np
import matplotlib.pylab as plt

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

# read data
data = np.loadtxt('ModelRocketData1.txt',skiprows=14)
tData = data[:,0]
TData = data[:,1]
tData = tData - 2.4
TData = np.abs(TData)
t=np.linspace(0,2.5,100)
T = []
for tt in t:
    TT = Thrust(tt)
    T.append(TT)
plt.plot(t,T)
plt.plot(tData,TData,linestyle='',marker='^')
plt.xlim(0,2.5)
plt.xlabel('t (s)',fontsize=14)
plt.ylabel('T (N)',fontsize=14)
plt.title('Thrust vs Time Run 1',fontsize=20,color='Brown')
plt.savefig('ThrustData1.png')
plt.show()