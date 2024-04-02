import numpy as np
from controlClasses.constants import ALQR,BLQR,QLQR,RLQR,PLQR
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
    k = -np.linalg.inv(RLQR)@BLQR.T@PLQR
    return k@delta

def valueFunctionLQR(system, control):
    systemCost = system.T@QLQR@system
    controlCost = control.T@RLQR@control
    return systemCost + controlCost
    