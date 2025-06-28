# import packages for communicating with the arduino, control the platform and 
# self made classes

import logging
import onnxruntime as ort
import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from controlClasses.algorithms import DSTA, LQR, ValueDNN
from controlClasses.constants import*
from controlClasses.functions import*
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise


def figure_creation(measures, positions, error, dotError, control_sgp, valueLQR, motor1):
    fig,ax = plt.subplots(2,2)

    fig.set_figheight(5)
    fig.set_figwidth(5)

    ax[0,0].plot(tiempo, measures[:,2], label = 'System 1')
    ax[0,0].plot(tiempo, positions[:,2], label = 'Reference 1')
    ax[0,0].legend()


    ax[0,1].plot(tiempo,error[:,0], label = 'Delta 1')
    ax[0,1].plot(tiempo, motor1.w1[1:], label = 'DSTA1 w1')
    ax[0,1].legend()


    ax[1,0].plot(tiempo,dotError[:,0], label = 'DSTA1 w2')
    ax[1,0].legend()


    ax[1,1].plot(tiempo,np.cumsum(valueLQR[:,0]), label = 'Value Function LQR')
    # ax[1,1].plot(tiempo,control_sgp[:,0], label = 'Proportional')
    # ax[1,1].plot(tiempo,controlD[:,0], label = 'Derivative')
    # ax[1,1].legend()

    print(np.cumsum(valueLQR[:,0])[-1])
    plt.show() 
# Close the serial connection





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
kp = np.array(
            [-100,-100,-100,-100,-100,-100]
            )

kd = np.array(
            [-10,-10,-10,-10,-10,-10]
            )


def sgp_main(kp,kd):
    # Conrol initialization
    control = np.zeros((1,6))+255
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

    control_sgp = np.zeros((time_steps,6))

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
            pdc,oc,delta = controlLQR.ocwPD(error[idx,:],dotError[idx,:],kp,kd)
            control = vControlBound((pdc+oc)[:,0])
            control_sgp[idx] = (pdc+oc)[:,0]

            #Control LQQR + DNN
            # optimalControl, delta = controlDNN.control(error[idx,:],dotError[idx,:])
            # control = vControlBound((optimalControl)[:,0])
            # control_sgp[idx] = (optimalControl)[:,0]

            valueLQR[idx,0] = valueFunctionLQR(delta,control)

            P = controlDNN.P[-1]
        
            plin[:,idx] = P.flatten()
            # Condicion de delta

            watcher = np.linalg.norm(delta)
            if watcher>100:
                print(watcher)
                print("Ahhhhhh")
                break


    # Obtain the time employed to run the algorithm
    toc = time.time() - tic
    print(toc/time_steps)
    print(valueLQR[-1,0])
    
    # Create a dictionary for storing the information
    motors = [motor1, motor2, motor3, motor4, motor5, motor6]
    dataAquired = saveData(measures,positions,control_sgp,motors,valueLQR)

    # Create a .csv file that containsthe information computed
    df = pd.DataFrame(dataAquired)
    figure_creation(measures, positions, error, dotError, control_sgp, valueLQR, motor1)
    return valueLQR[-1,0]
    
# df.to_csv(FILECSVPD)
# df.to_csv(FILECSVLQR)

sgp_main(kp = kp, kd = kd)


# RL
class EnvStewart(gym.Env):
    def __init__(self):
        super(EnvStewart, self).__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self.cost_function = 1
        self.reset()
    

    def step(self, action):
        # Send control value and received actuators position
        kp_delta, kd_delta = action[:6], action[6:]
        self.kp = np.clip(self.kp + kp_delta, -200, 200)
        self.kd = np.clip(self.kd + kd_delta, -50, 50)
        self.cost_function = sgp_main(kp = self.kp, kd = self.kd)
        obs = self._get_obs()
        reward = float(self._compute_reward())
        done = self._is_done(obs)
        truncated = self._is_truncated(obs)
        return obs, reward, done,truncated, {}

    def reset(self, seed=0):
        time.sleep(1)  # Wait for the system to stabilize
        self.kp = np.array(
            [-90,-90,-90,-90,-90,-90]
            )

        self.kd = np.array(
            [-10,-10,-10,-10,-10,-10]
            )
        return self._get_obs(), {"A":0}
    def _get_obs(self):
        return np.array([self.kp,self.kd], dtype=np.float32).flatten()
    def _compute_reward(self):
        return np.array([4000/self.cost_function],dtype=np.float32)
    def _is_done(self,obs):
        if (np.linalg.norm(obs) + self.cost_function) < 1000:
            return True
        return False
    def _is_truncated(self,obs):
        if (np.linalg.norm(obs) + self.cost_function) > 5000:
            return True
        return False
    def render(self):
        pass


# env = EnvStewart()
# check_env(env)



ser.close()

# Show 
