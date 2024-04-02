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
    [-250,0,0,0,0,0]
    ])

kd = np.array(
    [-100,0,0,0,0,0]
)

## LQR

ALQR = np.array([[-4,0.03],[0.75,-10]])
BLQR = np.array([[2],[0]])
QLQR = np.array([[500,0],[0,5]])
RLQR = np.array([[0.01]])
PLQR = np.array([[1, 0],[0,0.25]])


jLQR = controlD = np.zeros((time_steps,1))

# Storage

FILECSVPD = 'data\platformPD.csv'
FILECSVLQR = 'data\platformLQR.csv'