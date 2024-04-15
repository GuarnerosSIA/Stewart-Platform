import numpy as np
from controlClasses.constants import ALQR,BLQR2,QLQR2,RLQR2,PLQR2
# Serial communication
def control_bounds(x):
    if x >= 255:
        x = 255
    elif x <= -255:
        x = -255
    return x + 255

# LQR control

def LQR(delta):
    # The signs is reverse due to the sign of the output ports
    k = -np.linalg.inv(RLQR2)@BLQR2.T@PLQR2
    return k@delta

def valueFunctionLQR(system, control):
    systemCost = system.T@QLQR2@system
    controlCost = control.T@RLQR2@control
    return systemCost + controlCost
    