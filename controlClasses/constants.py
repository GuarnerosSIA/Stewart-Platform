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
    [-500,0,0,0,0,0]
    ])

nKp = np.reshape(kp,(6,1))

kd = np.array(
    [-5,0,0,0,0,0]
)

## LQR

ALQR = np.array([[-4,0.03],[0.75,-10]])
BLQR = np.array([[2],[0]])
QLQR = np.array([[500,0],[0,5]])
RLQR = np.array([[0.01]])
PLQR = np.array([[1, 0],[0,0.25]])

ALQR2 = np.zeros((6,6))
ALQR2[:3,3:] = np.eye(3)
ALQR2[3:,:3] = -np.eye(3)*500
ALQR2[3:,3:] = -np.eye(3)*5

BLQR2 = np.zeros((6,3))
BLQR2[3:,:] = np.eye(3)

QLQR2 = np.zeros((6,6))
QLQR2[:3,:3] = np.eye(3)*500
QLQR2[3:,3:] = np.eye(3)*5

RLQR2 = np.eye(3)*0.01

print(np.diag(ALQR2[3:,:3]))

PLQR2 = np.array([[96.4776,    0,  0, 0.8541,  0, 0],
                  [0,    96.4776,  0, 0,   0.8541, 0],
                  [0,    0,  96.4776, 0,  0, 0.8541],
                  [0.8541,    0,  0, 0.033,  0, 0],
                  [0,    0.8541,  0, 0,  0.033, 0],
                  [0,    0,  0.8541, 0,  0, 0.033],])

# PLQR2 = np.zeros((3,3))
# PLQR2[0,0] = 96.4776


# Storage

FILECSVPD = 'data\platformPD.csv'
FILECSVLQR = 'data\platformLQR.csv'