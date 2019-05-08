# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:18:49 2019

@author: Austin
"""
import numpy as np
import matplotlib.pylab as plt

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

# read data
data = np.loadtxt('ModelRocketData2.txt',skiprows=7)
tData = data[:,0]
TData = data[:,1]
tData = tData - 2.2
TData = np.abs(TData)
t=np.linspace(0,1.5,100)
T = []
for tt in t:
    TT = Thrust(tt)
    T.append(TT)
plt.plot(t,T)
plt.plot(tData,TData,linestyle='',marker='^')
plt.xlim(0,1)
plt.xlabel('t (s)',fontsize=14)
plt.ylabel('T (N)',fontsize=14)
plt.title('Thrust vs Time Run 2',fontsize=20,color='Brown')
plt.savefig('ThrustData2.png')
plt.show()