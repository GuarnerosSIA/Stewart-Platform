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

controlP = np.zeros((time_steps,6))
controlD = np.zeros((time_steps,6))
controlPD = np.zeros((time_steps,6))

valueLQR = np.zeros((time_steps,1))

# Measure of the time
tic = time.time()

# A class for the LQR is created in order to apply Linear Optimal Control
controlLQR = LQR(QLQR,RLQR,BLQR,ALQR,PLQR,0.01,0.0000001)
# controlDNN = ValueDNN(QLQR,RLQR,BLQR,ALQR,PLQR,alpha,beta,dt,w0,c)
print(PLQR)
#The Ricatti equation solution is computed
controlLQR.gainsComputation()
print(controlLQR.P)

for idx, idt in enumerate(tiempo):
    # Send control value and received actuators poition
    integers_to_send = [int(control[0,i]) for i in range(6)]
    
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
        controlProportional = np.multiply(error[idx,:],kp)
        controlDerivative = np.multiply(dotError[idx,:],kd)
               
        controlP[idx,:] = controlProportional
        controlD[idx,:] = controlDerivative
        controlPD[idx,:] = controlProportional + controlDerivative
        
        # storage the control in a variable in order to send it to the arduino

        # control = vControlBound(controlProportional + controlDerivative)
        
        # reshape the delta error to use it in the computation
        delta = np.reshape(np.concatenate((error[idx,:6],dotError[idx,:6])),(12,1))
        aux = vControlBound(controlLQR.opControl(delta)[:,0]+controlPD[idx,:6])
        control[0,:6] = [int(aux[i]) for i in range(6)]
        controlPD[idx,0] = control[0,0]
        valueLQR[idx,0] = valueFunctionLQR(delta,control[:,:6].T)
        # See the information send
        print(actuators)
    


# Obtain the time employed to run the algorithm
toc = time.time() - tic
print(toc)

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
    'LQR Integral value function':np.cumsum(valueLQR[:,0])
}

# Create a .csv file that containsthe information computed
df = pd.DataFrame(dataAquired)
df.to_csv(FILECSVPD)

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
# ax[1,1].plot(tiempo,controlP[:,0], label = 'Proportional')
# ax[1,1].plot(tiempo,controlD[:,0], label = 'Derivative')
ax[1,1].legend()


plt.show() 

# Close the serial connection
ser.close()