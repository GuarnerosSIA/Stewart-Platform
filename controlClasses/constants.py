import numpy as np 
import pandas as pd

# Default parameters

frecuencia = 0.05
dt = 0.01
expected_time = 10
time_steps = int(expected_time/dt)
tiempo = np.linspace(0,expected_time,time_steps, endpoint= False)

# Reference signal
# Read file
trajectory_df = pd.read_csv('data\\boat_trajectory.csv')

positions = np.zeros((time_steps,6))
positions[:,0] = 2.5*np.sin(2*np.pi*tiempo*frecuencia)+5
positions[:,1] = 2.5*np.sin(2*np.pi*tiempo*frecuencia)+5
positions[:,2] = 2.5*np.sin(2*np.pi*tiempo*frecuencia)+5
positions[:,3] = 2.5*np.sin(2*np.pi*tiempo*frecuencia)+5
positions[:,4] = 2.5*np.sin(2*np.pi*tiempo*frecuencia)+5
positions[:,5] = 2.5*np.sin(2*np.pi*tiempo*frecuencia)+5

# from file
positions[:,0] = trajectory_df['Leg_1'].values[:time_steps]
positions[:,1] = trajectory_df['Leg_2'].values[:time_steps]
positions[:,2] = trajectory_df['Leg_3'].values[:time_steps]
positions[:,3] = trajectory_df['Leg_4'].values[:time_steps]
positions[:,4] = trajectory_df['Leg_5'].values[:time_steps]
positions[:,5] = trajectory_df['Leg_6'].values[:time_steps]


# Control initialization



## PD control
kp = np.array([
    [-100,-100,-100,-100,-100,-100]
    ])

nKp = np.reshape(kp,(6,1))

kd = np.array(
    [-10,-10,-10,-10,-10,-10]
)

## LQR 6
ALQR = np.zeros((12,12))
ALQR[:6,6:] = np.eye(6)
ALQR[6:,:6] = -np.eye(6)*100
ALQR[6:,6:] = -np.eye(6)*10

BLQR = np.zeros((12,6))
BLQR[6:,:] = np.eye(6)

QLQR = np.zeros((12,12))
QLQR[:6,:6] = np.eye(6)*100
QLQR[6:,6:] = np.eye(6)*10

RLQR = np.eye(6)*0.01


PLQR = np.zeros((12,12))


# LQR 3
ALQR3 = np.zeros((12,12))
ALQR3[:6,6:] = np.eye(6)
ALQR3[6:,:6] = -np.eye(6)*20
ALQR3[6:,6:] = -np.eye(6)*19

BLQR3 = np.zeros((12,6))
BLQR3[6:,:] = np.eye(6)

QLQR3 = np.zeros((12,12))
QLQR3[:6,:6] = np.eye(6)*0.01
QLQR3[6:,6:] = np.eye(6)*0.01


RLQR3 = np.eye(6)*0.005


PLQR3 = np.zeros((12,12))
ppdiag =0.3
pddiag = 0.3
PLQR3[0,0]=ppdiag
PLQR3[1,1]=ppdiag
PLQR3[2,2]=ppdiag
PLQR3[3,3]=ppdiag
PLQR3[4,4]=ppdiag
PLQR3[5,5]=ppdiag
PLQR3[6,6]=pddiag
PLQR3[7,7]=pddiag
PLQR3[8,8]=pddiag
PLQR3[9,9]=pddiag
PLQR3[10,10]=pddiag
PLQR3[11,11]=pddiag

# Differential Neural Network
nStates = 12
nInputs = 6
nNeuronsV = 10

alpha = 0.6
beta =  0.6

w0 = np.random.random((nNeuronsV,1))*0.01
w0 = w0.astype(np.float32)
c = (np.random.random((nNeuronsV,nStates)).T-0.5)*0.001
# print(PLQR)
# print(PLQR2)
# Storage

FILECSVPD = 'data\platformPD_boat.csv'
FILECSVLQR = 'data\platformLQR_boat.csv'

# Reinforcement learning
actor_State = 'input_1'
critic_State = 'input_1'
critic_Action = 'input_2'