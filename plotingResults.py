from controlClasses.constants import*
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams


FILECSVPD = '.\Data\platformPD.csv'
FILECSVLQR = '.\Data\platformLQR.csv'

lqrControl = pd.read_csv(FILECSVPD)
dnnControl = pd.read_csv(FILECSVLQR)

rcParams.update({'font.family': 'Times New Roman',
                 'font.size':10,
                 'text.usetex':True})


for i in range(6):
    system = 'System '+str(i+1)
    reference = 'Reference '+str(i+1)
    file = './Tracking'+str(i+1)+'.png'

    plt.figure(figsize=(4,4),dpi=300)

    plt.axes([0.15, 0.15, 0.8, 0.8])

    ax = plt.gca()
    ax.plot(tiempo,lqrControl[reference],linewidth=10, color = 'black')
    ax.plot(tiempo, dnnControl[system], linewidth=7, color = 'tab:red')
    ax.plot(tiempo,lqrControl[system], linewidth=4, color = 'steelblue')
    
    ax.set_ylabel(r'Distance (cm)',size=12)
    ax.set_xlabel(r'Time (s)',size=12)
    ax.set_xlim(0,40)
    ax.set_ylim(2.5,15)
    ax.grid(True)
    plt.axes([0.3, 0.55, 0.6, 0.35])

    ax = plt.gca()
    
    ax.plot(tiempo,lqrControl[reference], label = 'Reference',linewidth=3, color = 'black')
    ax.plot(tiempo, dnnControl[system], label = 'LQR+DNN', linewidth=3, color = 'tab:red')
    ax.plot(tiempo,lqrControl[system], label = 'LQR+PD', linewidth=3, color = 'steelblue')
    
    ax.set_xlim(24,28)
    ax.set_ylim(6,7.1)
    ax.grid(True)
    ax.legend(fontsize=10)

    # plt.tight_layout()
    plt.savefig(file)


for i in range(6):
    control = 'Control '+str(i+1)
    file = './Control'+str(i+1)+'.png'

    plt.figure(figsize=(4,4),dpi=300)

    plt.axes([0.2, 0.15, 0.75, 0.8])

    ax = plt.gca()
    ax.plot(tiempo,dnnControl[control], label = 'LQR+DNN', linewidth=2, color = 'tab:red')
    ax.plot(tiempo,lqrControl[control], label = 'LQR+PD', linewidth=2, color = 'steelblue')

    ax.set_ylabel(r'PWM',size=12)
    ax.set_xlim(0,40)
    ax.set_ylim(-150,500)
    ax.grid(True)
    ax.set_xlabel(r'Time (s)',size=12)


    plt.axes([0.35, 0.55, 0.5, 0.35])

    ax = plt.gca()
    ax.plot(tiempo,dnnControl[control], label = 'LQR+DNN', linewidth=2, color = 'tab:red')
    ax.plot(tiempo,lqrControl[control], label = 'LQR+PD', linewidth=2, color = 'steelblue')

    ax.set_xlim(24,28)
    ax.set_ylim(-150,100)
    ax.grid(True)
    ax.legend()

    plt.savefig(file)



plt.figure(figsize=(4,4),dpi=300)

plt.axes([0.2, 0.15, 0.75, 0.8])

ax = plt.gca()

ax.plot(tiempo,lqrControl['LQR Integral value function'], label = 'LQR+PD', 
        linewidth=5,linestyle='-', color = 'steelblue')
ax.plot(tiempo, dnnControl['LQR Integral value function'], label = 'LQR+DNN',
         linewidth=5,linestyle='-', color = 'tab:red')
ax.set_ylabel(r'$J_{t}(t_{0},\Delta_{0};u_{op})$',size=12)
ax.set_xlabel(r'Time (s)')
ax.set_xlim(0,40)
ax.set_ylim(0,3.5e7)
ax.grid(True)

plt.axes([0.35, 0.55, 0.5, 0.35])

ax = plt.gca()

ax.plot(tiempo,lqrControl['LQR Integral value function'], label = 'LQR+PD', 
        linewidth=5,linestyle='-', color = 'steelblue')
ax.plot(tiempo, dnnControl['LQR Integral value function'], label = 'LQR+DNN',
         linewidth=5,linestyle='-', color = 'tab:red')
ax.set_xlim(36,40)
ax.set_ylim(1.45e7,1.7e7)
ax.grid(True)
ax.legend(fontsize=10)


plt.savefig('./functional.png')
