from controlClasses.constants import*
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

FILECSVPD = '.\Data\platformPD.csv'
FILECSVLQR = '.\Data\platformLQR.csv'

lqrControl = pd.read_csv(FILECSVPD)
dnnControl = pd.read_csv(FILECSVLQR)


for i in range(6):
    fig,ax = plt.subplots(2,1)

    fig.set_figheight(5)
    fig.set_figwidth(10)

    system = 'System '+str(i+1)
    reference = 'Reference '+str(i+1)
    control = 'Control '+str(i+1)

    file = './Actuator'+str(i+1)+'.png'

    ax[0].plot(tiempo,lqrControl[system], label = 'LQR+PD')
    ax[0].plot(tiempo,lqrControl[reference], label = 'Reference')
    ax[0].plot(tiempo, dnnControl[system], label = 'LQR+DNN')
    ax[0].legend()

    ax[1].plot(tiempo,dnnControl[control], label = 'LQR+DNN')
    ax[1].plot(tiempo,lqrControl[control], label = 'LQR+PD')
    
    ax[1].legend()

    # ax[1,1].plot(tiempo,np.cumsum(valueLQR[:,0]), label = 'Value Function LQR')
    fig.savefig(file)


fig,ax = plt.subplots(1,1)

fig.set_figheight(5)
fig.set_figwidth(10)

ax.plot(tiempo,lqrControl['LQR Integral value function'], label = 'LQR+PD')
ax.plot(tiempo, dnnControl['LQR Integral value function'], label = 'LQR+DNN')
ax.legend()

fig.savefig('./functional.png')
