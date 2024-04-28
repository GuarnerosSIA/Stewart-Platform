import numpy as np
from controlClasses.algorithms import LQR



ALQR = np.zeros((6,6))
ALQR[:3,3:] = np.eye(3)
ALQR[3:,:3] = -np.eye(3)*500
ALQR[3:,3:] = -np.eye(3)*5

BLQR = np.zeros((6,3))
BLQR[3:,:] = np.eye(3)

QLQR = np.zeros((6,6))
QLQR[:3,:3] = np.eye(3)*500
QLQR[3:,3:] = np.eye(3)*5

RLQR = np.eye(3)*0.01

print(np.diag(ALQR[3:,:3]))

dt = 0.01
T = 5
time = int(T/dt)

PLQR = np.zeros((6,6,time+1))

controlLQR = LQR(QLQR,RLQR,BLQR,ALQR,PLQR[:,:,0],dt,0.0001)


for i in range(time):
    PLQR[:,:,i+1] = dt*-1*(-PLQR[:,:,i]@ALQR-ALQR.T@PLQR[:,:,i]-QLQR + PLQR[:,:,i]@BLQR@np.linalg.inv(RLQR)@BLQR.T@PLQR[:,:,i])+PLQR[:,:,i]
    
controlLQR.gainsComputation()

print(PLQR[0,0,-3])
print(controlLQR.P)
print(controlLQR.K[0])
print(controlLQR.K[-1])