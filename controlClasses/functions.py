import numpy as np
from controlClasses.constants import ALQR,BLQR,QLQR,RLQR,PLQR
import serial

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
    

def DoPri45Step(f,t,x,h):
    
    k1 = f(t,x)
    k2 = f(t + 1./5*h, x + h*(1./5*k1) )
    k3 = f(t + 3./10*h, x + h*(3./40*k1 + 9./40*k2) )
    k4 = f(t + 4./5*h, x + h*(44./45*k1 - 56./15*k2 + 32./9*k3) )
    k5 = f(t + 8./9*h, x + h*(19372./6561*k1 - 25360./2187*k2 + 64448./6561*k3 - 212./729*k4) )
    k6 = f(t + h, x + h*(9017./3168*k1 - 355./33*k2 + 46732./5247*k3 + 49./176*k4 - 5103./18656*k5) )

    v5 = 35./384*k1 + 500./1113*k3 + 125./192*k4 - 2187./6784*k5 + 11./84*k6
    k7 = f(t + h, x + h*v5)
    v4 = 5179./57600*k1 + 7571./16695*k3 + 393./640*k4 - 92097./339200*k5 + 187./2100*k6 + 1./40*k7
    
    return v4,v5


def sendReceive(integers2Send, serialObject):
    data_to_send = ','.join(map(str, integers2Send)) + '\n'
    serialObject.write(data_to_send.encode('utf-8'))
    data_received = serialObject.readline()
    return data_received.decode('utf-8')