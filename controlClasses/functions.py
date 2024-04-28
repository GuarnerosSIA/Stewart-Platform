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



def valueFunctionLQR(system, control):
    systemCost = system.T@QLQR@system
    controlCost = control.T@RLQR@control
    return systemCost + controlCost
    