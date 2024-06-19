import numpy as np

# Default parameters

frecuencia = 0.1
dt = 0.01
expected_time = 10
time_steps = int(expected_time/dt)
tiempo = np.linspace(0,expected_time,time_steps, endpoint= False)

# Reference signal

positions = np.zeros((time_steps,6))
positions[:,0] = 2*np.sin(2*np.pi*tiempo*frecuencia)+5
positions[:,1] = 2*np.cos(2*np.pi*tiempo*frecuencia)+5
positions[:,2] = 2*np.sin(2*np.pi*tiempo*frecuencia)+5
positions[:,3] = 2*np.cos(2*np.pi*tiempo*frecuencia)+5
positions[:,4] = 2*np.sin(2*np.pi*tiempo*frecuencia)+5
positions[:,5] = 2*np.cos(2*np.pi*tiempo*frecuencia)+5


# Control initialization

control = np.zeros((1,6))+255

## PD control
kp = np.array([
    [-500,-500,-500,-500,-500,-500]
    ])

nKp = np.reshape(kp,(6,1))

kd = np.array(
    [-5,-5,-5,-5,-5,-5]
)

## LQR 6
ALQR = np.zeros((12,12))
ALQR[:6,6:] = np.eye(6)
ALQR[6:,:6] = -np.eye(6)*500
ALQR[6:,6:] = -np.eye(6)*5

BLQR = np.zeros((12,6))
BLQR[6:,:] = np.eye(6)

QLQR = np.zeros((12,12))
QLQR[:6,:6] = np.eye(6)*500
QLQR[6:,6:] = np.eye(6)*5

RLQR = np.eye(6)*0.01


PLQR = np.zeros((12,12))

# LQR 1
ALQR1 = np.zeros((2,2))
ALQR1[:1,1:] = np.eye(1)
ALQR1[1:,:1] = -np.eye(1)*500
ALQR1[1:,1:] = -np.eye(1)*5

BLQR1 = np.zeros((2,1))
BLQR1[1:,:] = np.eye(1)

QLQR1 = np.zeros((1,1))
QLQR1[:1,:1] = np.eye(1)*500
QLQR1[1:,1:] = np.eye(1)*5

RLQR1 = np.eye(1)*0.01


PLQR1 = np.zeros((2,2))

# Differential Neural Network
nStates = 12
nInputs = 6
nNeuronsV = 5

alpha = 0.5
beta =  0.5

w0 = np.random.random((nNeuronsV,1))*10
c = np.random.random((nNeuronsV,nStates)).T*0.01
# print(PLQR)
# print(PLQR2)
# Storage

FILECSVPD = 'data\platformPD.csv'
FILECSVLQR = 'data\platformLQR.csv'