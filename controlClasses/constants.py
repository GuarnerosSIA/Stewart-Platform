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



# LQR 1
ALQR1 = np.zeros((2,2))
ALQR1[:1,1:] = np.eye(1)
ALQR1[1:,:1] = -np.eye(1)*100
ALQR1[1:,1:] = -np.eye(1)*10

BLQR1 = np.zeros((2,1))
BLQR1[1:,:] = np.eye(1)

QLQR1 = np.zeros((2,2))
QLQR1[:1,:1] = np.eye(1)*0.05
QLQR1[1:,1:] = np.eye(1)*0.05

RLQR1 = np.eye(1)*0.001


PLQR1 = np.zeros((2,2))
PLQR1[0,0]=5
PLQR1[1,1]=5



# LQR 2
ALQR2 = np.zeros((4,4))
ALQR2[:2,2:] = np.eye(2)
ALQR2[2:,:2] = -np.eye(2)*15
ALQR2[2:,2:] = -np.eye(2)*4

BLQR2 = np.zeros((4,2))
BLQR2[2:,:] = np.eye(2)

QLQR2 = np.zeros((4,4))
QLQR2[:2,:2] = np.eye(2)*0.01
QLQR2[2:,2:] = np.eye(2)*0.01

RLQR2 = np.eye(2)*0.001


PLQR2 = np.zeros((4,4))
PLQR2[0,0]=1
PLQR2[1,1]=1
PLQR2[2,2]=1
PLQR2[3,3]=1


# LQR 3
ALQR3 = np.zeros((12,12))
ALQR3[:6,6:] = np.eye(6)
ALQR3[6:,:6] = -np.eye(6)*15
ALQR3[6:,6:] = -np.eye(6)*4

BLQR3 = np.zeros((12,6))
BLQR3[6:,:] = np.eye(6)

QLQR3 = np.zeros((12,12))
QLQR3[:6,:6] = np.eye(6)*0.01
QLQR3[6:,6:] = np.eye(6)*0.01

RLQR3 = np.eye(6)*0.001


PLQR3 = np.zeros((12,12))
PLQR3[0,0]=5
PLQR3[1,1]=5
PLQR3[2,2]=5
PLQR3[3,3]=5
PLQR3[4,4]=5
PLQR3[5,5]=5
PLQR3[6,6]=5
PLQR3[7,7]=5
PLQR3[8,8]=5
PLQR3[9,9]=5
PLQR3[10,10]=5
PLQR3[11,11]=5

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