import numpy as np
import matplotlib.pyplot as plt
from controlClasses.algorithms import ValueDNN
from controlClasses.functions import DoPri45Step
from controlClasses.constants import*
from matplotlib import rcParams
rcParams.update({'font.family': 'Times New Roman',
                 'font.size':10,
                 'text.usetex':True})

steps = 4000
t = np.linspace(0,40,steps)
Pint = np.zeros((steps,144))
time = []

def riccati(t,p):
    # p = p.reshape((2,2))
    pa = p@ALQR3
    ap = ALQR3.T@p
    pb = p@BLQR3
    q = QLQR3
    rInv = np.linalg.inv(RLQR3)
    bp = BLQR3.T@p
    return -1*(-pa-ap-q + pb@rInv@bp)
    

Pint[0,:] = PLQR3.flatten()
for i in range(steps-1):
    h = t[i+1]-t[i]
    v4, v5 = DoPri45Step(riccati,t[i],PLQR3,h)
    newP = PLQR3 + h*v5
    
    
    # p11.append(newP[0][0])
    # p21.append(newP[1][0])
    # p12.append(newP[0][1])
    # p22.append(newP[1][1])
    PLQR3 = newP
    Pint[i+1,:] = PLQR3.flatten()


print(PLQR3)

plt.figure(figsize=(4,4),dpi=300)

plt.axes([0.15, 0.15, 0.8, 0.8])

ax = plt.gca()


ax.plot(t,Pint[:,0], label = 'LQR+PD', 
        linewidth=5,linestyle='-',color='steelblue')
ax.plot(t,Pint[:,6], label = 'LQR+PD', 
        linewidth=5,linestyle='-',color='tab:red')
ax.plot(t,Pint[:,78], label = 'LQR+PD', 
        linewidth=5,linestyle='-',color='seagreen')
# ax.plot(time, dnnControl['LQR Integral value function'], label = 'LQR+DNN',
#          linewidth=5,linestyle='-', color = 'tab:red')
# ax.set_xlim(36,40)
# ax.set_ylim(1.45e7,1.7e7)
ax.set_ylabel(r'Values from $P_{t}$',size=12)
ax.set_xlabel(r'Time (s)',size=12)
ax.grid(True)
# ax.legend(fontsize=10)

plt.axes([0.35, 0.35, 0.5, 0.55])

ax = plt.gca()


ax.plot(t,Pint[:,0], label = r'$P_{i,i}$', 
        linewidth=5,linestyle='-',color='steelblue')
ax.plot(t,Pint[:,6], label = r'$P_{i,j}=P_{j,i}$', 
        linewidth=5,linestyle='-',color='tab:red')
ax.plot(t,Pint[:,78], label = r'$P_{j,j}$', 
        linewidth=5,linestyle='-',color='seagreen')
# ax.plot(time, dnnControl['LQR Integral value function'], label = 'LQR+DNN',
#          linewidth=5,linestyle='-', color = 'tab:red')
ax.set_xlim(0,1)
# ax.set_ylim(1.45e7,1.7e7)
ax.grid(True)
ax.legend(fontsize=10)


plt.savefig('./Riccati.png')
