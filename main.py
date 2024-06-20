# import packages for communicating with the arduino, control the platform and 
# self made classes

import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from controlClasses.algorithms import DSTA, LQR, ValueDNN
from controlClasses.constants import*
from controlClasses.functions import*

# Start the serial port 

ser = serial.Serial('COM10', 250000)
time.sleep(3)

# create a scalar function into a function for vectors
vControlBound = np.vectorize(control_bounds)

# A class for the Discrete Super Stwisting Algorithm is 
# created for each of the motors used

motor1 = DSTA(dt,6,5,0.999, 0.999,w1=0,w2=0)
motor2 = DSTA(dt,6,5,0.999, 0.999,w1=0,w2=0)
motor3 = DSTA(dt,6,5,0.999, 0.999,w1=0,w2=0)
motor4 = DSTA(dt,6,5,0.999, 0.999,w1=0,w2=0)
motor5 = DSTA(dt,6,5,0.999, 0.999,w1=0,w2=0)
motor6 = DSTA(dt,6,5,0.999, 0.999,w1=0,w2=0)

# Create arrays to store the information computed and obtained

measures = np.zeros((time_steps,6))
dotMeasures = np.zeros((time_steps,6))

error = np.zeros((time_steps,6))
dotError = np.zeros((time_steps,6))

controlKillMe = np.zeros((time_steps,6))

valueLQR = np.zeros((time_steps,1))
plin = np.zeros((nStates*nStates,time_steps))


# Measure of the time
tic = time.time()

# A class for the LQR is created in order to apply Linear Optimal Control
controlLQR = LQR(QLQR3,RLQR3,BLQR3,ALQR3,PLQR3,0.01,0.0000001)
controlDNN = ValueDNN(QLQR3,RLQR3,BLQR3,ALQR3,PLQR3,alpha,beta,dt,w0,c)
#The Ricatti equation solution is computed
controlLQR.gainsComputation()
print(ALQR3)

for idx, idt in enumerate(tiempo):
    # Send control value and received actuators poition
    integers_to_send = control.astype(int).tolist()
    
    actuators = sendReceive(integers_to_send,ser)
    # Evaluate if the received information was correct
    if actuators[0]=='A':
        # Separate the information obtained 
        act_sep = actuators[1:].replace('\r\n','').split(',')
        measures[idx,:] = np.array([float(item) for item in act_sep])
        # Compute the trajectory tracking error
        deltas = positions[idx,:] - measures[idx,:]
        error[idx,:] = deltas
        # Compute the error derivative with the STA
        dotError[idx,0] = motor1.derivative(error[idx,0])
        dotError[idx,1] = motor2.derivative(error[idx,1])
        dotError[idx,2] = motor3.derivative(error[idx,2])
        dotError[idx,3] = motor4.derivative(error[idx,3])
        dotError[idx,4] = motor5.derivative(error[idx,4])
        dotError[idx,5] = motor6.derivative(error[idx,5])
        # Calculate the proportional and derivative control for the PD
        
        #Control LQR + PD
        pdc,oc,delta = controlLQR.ocwPD(error[idx,:],dotError[idx,:],kp,kd)
        control = vControlBound((pdc+oc)[:,0])
        controlKillMe[idx] = (pdc+oc)[:,0]
        
        #Control LQQR + DNN
        # optimalControl, delta = controlDNN.control(error[idx,:],dotError[idx,:])
        # control = vControlBound((optimalControl)[:,0])
        # controlKillMe[idx] = (optimalControl)[:,0]

        valueLQR[idx,0] = valueFunctionLQR(delta,control)

        P = controlDNN.P[-1]
        
        plin[:,idx] = P.flatten()
        
        # See the information send
        # print(control)
        # print(integers_to_send)
    


# Obtain the time employed to run the algorithm
toc = time.time() - tic
print(toc/time_steps)

# Create a dictionary for storing the information
dataAquired = {
    'System 1':measures[:,0], 'System 2':measures[:,1], 'System 3':measures[:,2],
    'System 4':measures[:,3], 'System 5':measures[:,4], 'System 6':measures[:,5],
    'Reference 1':positions[:,0], 'Reference 2':positions[:,1], 'Reference 3':positions[:,2],
    'Reference 4':positions[:,3], 'Reference 5':positions[:,4], 'Reference 6':positions[:,5],
    
    'M1 STA':motor1.w1[1:],
    'DM1 STA':motor1.w2[1:],
    'M2 STA':motor2.w1[1:],
    'DM2 STA':motor2.w2[1:],
    'M3 STA':motor3.w1[1:],
    'DM3 STA':motor3.w2[1:],
    'M4 STA':motor4.w1[1:],
    'DM4 STA':motor4.w2[1:],
    'M5 STA':motor5.w1[1:],
    'DM5 STA':motor5.w2[1:],
    'M6 STA':motor6.w1[1:],
    'DM6 STA':motor6.w2[1:],

    'LQR Value function':valueLQR[:,0],
    'LQR Integral value function':np.cumsum(valueLQR[:,0]),

    'Control 1':controlKillMe[:,0],
    'Control 2':controlKillMe[:,1],
    'Control 3':controlKillMe[:,2],
    'Control 4':controlKillMe[:,3],
    'Control 5':controlKillMe[:,4],
    'Control 6':controlKillMe[:,5]
}

# Create a .csv file that containsthe information computed
df = pd.DataFrame(dataAquired)

df.to_csv(FILECSVPD)
# df.to_csv(FILECSVLQR)


# Show Figures
fig,ax = plt.subplots(2,2)

fig.set_figheight(5)
fig.set_figwidth(5)

ax[0,0].plot(tiempo, measures[:,0], label = 'System 1')
ax[0,0].plot(tiempo, positions[:,0], label = 'Reference 1')
ax[0,0].legend()


ax[0,1].plot(tiempo,error[:,0], label = 'Delta 1')
ax[0,1].plot(tiempo, motor1.w1[1:], label = 'DSTA1 w1')
ax[0,1].legend()


ax[1,0].plot(tiempo,dotError[:,0], label = 'DSTA1 w2')
ax[1,0].legend()


ax[1,1].plot(tiempo,np.cumsum(valueLQR[:,0]), label = 'Value Function LQR')
# ax[1,1].plot(tiempo,controlKillMe[:,0], label = 'Proportional')
# ax[1,1].plot(tiempo,controlD[:,0], label = 'Derivative')
# ax[1,1].legend()


plt.show() 

# Close the serial connection
ser.close()