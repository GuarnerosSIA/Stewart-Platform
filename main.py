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
ser = serial.Serial('COM3', 250000)
time.sleep(3)

# create a scalar function into a function for vectors
vControlBound = np.vectorize(control_bounds)

rlIterations = 1
actorNN,criticNN = loadRLNN()

np.random.seed(1)
w0 = np.random.random((nNeuronsV,1))*1
w0 = w0.astype(np.float32)


for i in range(rlIterations):
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
            # pdc,oc,delta = controlLQR.ocwPD(error[idx,:],dotError[idx,:],kp,kd)
            # control = vControlBound((pdc+oc)[:,0])
            # controlKillMe[idx] = (pdc+oc)[:,0]
        
            #Control LQQR + DNN
            optimalControl, delta = controlDNN.control(error[idx,:],dotError[idx,:])
            control = vControlBound((optimalControl)[:,0])
            controlKillMe[idx] = (optimalControl)[:,0]

            valueLQR[idx,0] = valueFunctionLQR(delta,control)

            P = controlDNN.P[-1]
        
            plin[:,idx] = P.flatten()
            # Condicion de delta
            
            watcher = np.linalg.norm(delta)
            if watcher>7:
                print(watcher)
                print("Ahhhhhh")
                break




    #RL Neural network    
    
    nnInputs = w0.T*0.005

    actionNN = actorNN.run(None,{actor_State:nnInputs})[0]
    actionRL = actionNN[0]
    qValue = criticNN.run(None,{critic_State:nnInputs,
                                critic_Action:actionNN})
    
    
    w0 = w0 + np.reshape(actionRL,(10,1))

    # Obtain the time employed to run the algorithm
    toc = time.time() - tic
    print(toc/time_steps)
    print(valueLQR[-1,0])
    print(qValue)




    # Create a dictionary for storing the information
    motors = [motor1, motor2, motor3, motor4, motor5, motor6]
    dataAquired = saveData(measures,positions,controlKillMe,motors,valueLQR)

    # Create a .csv file that containsthe information computed
    df = pd.DataFrame(dataAquired)
    time.sleep(10)

# df.to_csv(FILECSVPD)
# df.to_csv(FILECSVLQR)

ser.close()
# Show 


fig,ax = plt.subplots(2,2)

fig.set_figheight(5)
fig.set_figwidth(5)

ax[0,0].plot(tiempo, measures[:,1], label = 'System 1')
ax[0,0].plot(tiempo, positions[:,1], label = 'Reference 1')
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
