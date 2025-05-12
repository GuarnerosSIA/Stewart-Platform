from controlClasses.constants import*
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams


frecuencia = 0.05
dt = 0.01
expected_time = 40
time_steps = int(expected_time/dt)
tiempo = np.linspace(0,expected_time,time_steps, endpoint= False)


FILECSVPD = '.\Data\platformPD.csv'
FILECSVLQR = '.\Data\platformLQR.csv'

lqrControl = pd.read_csv(FILECSVPD)
dnnControl = pd.read_csv(FILECSVLQR)

rcParams.update({'font.family': 'Times New Roman',
                 'font.size':33,
                 'text.usetex':True})

onyx = (0.2235, 0.2431, 0.2549)
cinnabar = (0.9137, 0.3098, 0.2157)
steelblue = (0.2471, 0.5333, 0.7725)


for i in range(6):
    system = 'System '+str(i+1)
    reference = 'Reference '+str(i+1)
    file = './Tracking_Corrected'+str(i+1)+'.png'

    plt.figure(figsize=(4,2),dpi=100)

    plt.axes([0.15, 0.15, 0.8, 0.8])

    ax = plt.gca()
    ax.plot(tiempo,lqrControl[reference], label = 'Reference',linewidth=15, color = onyx)
    ax.plot(tiempo, dnnControl[system], label = 'V-DNN', linewidth=10, color = steelblue,
            linestyle=':')
    ax.plot(tiempo,lqrControl[system], label = 'LQR+PD', linewidth=10, color = cinnabar,
            linestyle='-.')
    
    ax.set_ylabel(r'Distance (cm)',size=33)
    ax.set_xlabel(r'Time (s)',size=33)
    ax.set_xlim(0,40)
    ax.set_ylim(2.5,15)
    ax.grid(True)
    ax.legend(fontsize=33)
    plt.axes([0.3, 0.55, 0.3, 0.35])

    ax = plt.gca()
    
    ax.plot(tiempo,lqrControl[reference],linewidth=15, color = onyx)
    ax.plot(tiempo, dnnControl[system], linewidth=10, color = steelblue,linestyle=':')
    ax.plot(tiempo,lqrControl[system], linewidth=10, color = cinnabar,linestyle='-.')
    
    ax.set_xlim(24,28)
    ax.set_ylim(6,7.1)
    ax.grid(True)
    

    # plt.show()
    # plt.tight_layout()
    # plt.savefig(file)


for i in range(6):
    control = 'Control '+str(i+1)
    file = './Control'+str(i+1)+'.png'

    plt.figure(figsize=(4,2),dpi=100)

    plt.axes([0.2, 0.15, 0.75, 0.8])

    ax = plt.gca()
    ax.plot(tiempo,dnnControl[control], label = 'V-DNN', linewidth=10, color = steelblue,
            linestyle=':')
    ax.plot(tiempo,lqrControl[control], label = 'LQR+PD', linewidth=10, color = cinnabar,
            linestyle='-.')

    ax.set_ylabel(r'PWM',size=33)
    ax.set_xlim(0,40)
    ax.set_ylim(-150,500)
    ax.grid(True)
    ax.legend()
    ax.set_xlabel(r'Time (s)',size=33)


    plt.axes([0.35, 0.55, 0.3, 0.35])

    ax = plt.gca()
    ax.plot(tiempo,dnnControl[control], label = 'V-DNN', linewidth=10, color = steelblue,
            linestyle=':')
    ax.plot(tiempo,lqrControl[control], label = 'LQR+PD', linewidth=10, color = cinnabar,
            linestyle='-.')

    ax.set_xlim(24,28)
    ax.set_ylim(-150,100)
    ax.grid(True)
    plt.show()

    # plt.savefig(file)



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


# plt.savefig('./functional.png')
