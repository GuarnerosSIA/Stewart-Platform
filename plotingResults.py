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
    fig,ax = plt.subplots(2,1, dpi=300)

    fig.set_figheight(3)
    fig.set_figwidth(7)

    system = 'System '+str(i+1)
    reference = 'Reference '+str(i+1)
    control = 'Control '+str(i+1)

    file = './Actuator'+str(i+1)+'.png'

    ax[0].plot(tiempo,lqrControl[reference], label = 'Reference',linewidth=10, color = 'black')
    ax[0].plot(tiempo, dnnControl[system], linewidth=7, color = 'tab:red')
    ax[0].plot(tiempo,lqrControl[system], linewidth=4, color = 'steelblue')
    
    ax[0].set_ylabel(r'Distance (cm)',size=12)
    ax[0].set_xlim(0,40)
    ax[0].set_ylim(2.5,8)
    ax[0].grid(True)
    ax[0].legend(fontsize=10)

    ax[1].plot(tiempo,dnnControl[control], label = 'LQR+DNN', linewidth=2, color = 'tab:red')
    ax[1].plot(tiempo,lqrControl[control], label = 'LQR+PD', linewidth=2, color = 'steelblue')

    ax[1].set_ylabel(r'PWM',size=12)
    ax[1].set_xlim(0,40)
    ax[1].set_ylim(-200,350)
    ax[1].grid(True)
    ax[1].legend()

    ax[1].set_xlabel(r'Time (s)',size=12)

    plt.tight_layout()
    fig.savefig(file)


fig,ax = plt.subplots(1,1)

fig.set_figheight(3)
fig.set_figwidth(7)

ax.plot(tiempo,lqrControl['LQR Integral value function'], label = 'LQR+PD', 
        linewidth=5,linestyle='-', color = 'steelblue')
ax.plot(tiempo, dnnControl['LQR Integral value function'], label = 'LQR+DNN',
         linewidth=5,linestyle='-', color = 'tab:red',alpha=0.7)
ax.set_ylabel(r'Functional $J_{t}$',size=12)
ax.set_xlim(0,40)
# ax.set_ylim(2.5,8)
ax.grid(True)
ax.legend(fontsize=10)


plt.tight_layout()


fig.savefig('./functional.png')
