import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from controlClasses.algorithms import DSTA
from controlClasses.constants import*
from controlClasses.functions import*

ser = serial.Serial('COM10', 250000)
time.sleep(3)

vControlBound = np.vectorize(control_bounds)

motor1 = DSTA(dt,6,4,0.99, 0.99,w1=0,w2=0)
motor2 = DSTA(dt,6,5,0.9, 0.99,w1=0,w2=0)
motor3 = DSTA(dt,6,5,0.9, 0.99,w1=0,w2=0)
motor4 = DSTA(dt,6,5,0.9, 0.99,w1=0,w2=0)
motor5 = DSTA(dt,6,5,0.9, 0.99,w1=0,w2=0)
motor6 = DSTA(dt,6,5,0.9, 0.99,w1=0,w2=0)

#Data adquisition
measures = np.zeros((time_steps,6))
dotMeasures = np.zeros((time_steps,6))

error = np.zeros((time_steps,6))
dotError = np.zeros((time_steps,6))

controlP = np.zeros((time_steps,6))
controlD = np.zeros((time_steps,6))
controlPD = np.zeros((time_steps,6))

tic = time.time()



for idx, idt in enumerate(tiempo):
# Send integers to Arduino
    integers_to_send = [int(control[0,0]),
                        int(control[0,1]), 
                        int(control[0,2]),
                        int(control[0,3]),
                        int(control[0,4]),
                        int(control[0,5])]
    data_to_send = ','.join(map(str, integers_to_send)) + '\n'
    print(data_to_send)
    ser.write(data_to_send.encode('utf-8'))
    A = ser.readline()
    actuators = A.decode('utf-8')
    if actuators[0]=='A':
        act_sep = actuators[1:].replace('\r\n','').split(',')
        measurements = np.array([float(item) for item in act_sep])

        measures[idx,:] = measurements
        deltas = positions[idx,:] - measurements
        
        error[idx,:] = deltas
        
        dotError[idx,0] = motor1.derivative(error[idx,0])
        dotError[idx,1] = motor2.derivative(error[idx,1])
        dotError[idx,2] = motor3.derivative(error[idx,2])
        dotError[idx,3] = motor4.derivative(error[idx,3])
        dotError[idx,4] = motor5.derivative(error[idx,4])
        dotError[idx,5] = motor6.derivative(error[idx,5])
        
        controlProportional = np.multiply(error[idx,:],kp)
        controlDerivative = np.multiply(dotError[idx,:],kd)
        
        controlP[idx,:] = controlProportional
        controlD[idx,:] = controlDerivative
        controlPD[idx,:] = controlProportional + controlDerivative
        
        # control = vControlBound(controlProportional + controlDerivative)
        delta1 = np.array([[error[idx,0]],[dotError[idx,0]]])
        control[0,0] = int(vControlBound(LQR(delta1))[0,0])
        controlPD[idx,0] = control[0,0]
        # print(deltas)
    time.sleep(0)     


# Close the serial connection
toc = time.time() - tic
print(toc)

dataAquired = {
    'System 1':measures[:,0], 'System 2':measures[:,1], 'System 3':measures[:,2],
    'System 4':measures[:,3], 'System 5':measures[:,4], 'System 6':measures[:,5],
    'Reference 1':positions[:,0], 'Reference 2':positions[:,1], 'Reference 3':positions[:,2],
    'Reference 4':positions[:,3], 'Reference 5':positions[:,4], 'Reference 6':positions[:,5],
    'M1 STA':motor1.w1[1:],
    'DM1 STA':motor1.w2[1:]
}

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


ax[1,1].plot(tiempo,controlPD[:,0], label = 'Control')
ax[1,1].plot(tiempo,controlP[:,0], label = 'Proportional')
ax[1,1].plot(tiempo,controlD[:,0], label = 'Derivative')
ax[1,1].legend()


plt.show() 


# fig,ax = plt.subplots(2,3)

# fig.set_figheight(5)
# fig.set_figwidth(5)

# ax[0,0].plot(tiempo, measures[:,0])
# ax[0,0].plot(tiempo, positions[:,0])
# ax[0,0].plot(tiempo, dotError[:,0])

# ax[0,1].plot(tiempo, measures[:,1])
# ax[0,1].plot(tiempo, positions[:,1])

# ax[0,2].plot(tiempo, measures[:,2])
# ax[0,2].plot(tiempo, positions[:,2])

# ax[1,0].plot(tiempo, measures[:,3])
# ax[1,0].plot(tiempo, positions[:,3])

# ax[1,1].plot(tiempo, measures[:,4])
# ax[1,1].plot(tiempo, positions[:,4])

# ax[1,2].plot(tiempo, measures[:,5])
# ax[1,2].plot(tiempo, positions[:,5])

# plt.show() 
# # storing the information in a CSV file


ser.close()