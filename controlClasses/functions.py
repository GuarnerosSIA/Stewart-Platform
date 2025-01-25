import numpy as np
from controlClasses.constants import ALQR,BLQR,QLQR,RLQR,PLQR
import onnx
import onnxruntime as ort

# Serial communication
def control_bounds(x):
    if x >= 255:
        x = 255
    elif x <= -255:
        x = -255
    return x + 255

# LQR control



def valueFunctionLQR(system, control):
    control = control.reshape((-1,1))
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

def saveData(measures, positions, control, motors, functional):
    motor1 = motors[0]
    motor2 = motors[1]
    motor3 = motors[2]
    motor4 = motors[3]
    motor5 = motors[4]
    motor6 = motors[5]
    dataAquired = {
    'System 1':measures[:,0], 'System 2':measures[:,1], 'System 3':measures[:,2],
    'System 4':measures[:,3], 'System 5':measures[:,4], 'System 6':measures[:,5],
    'Reference 1':positions[:,0], 'Reference 2':positions[:,1], 'Reference 3':positions[:,2],
    'Reference 4':positions[:,3], 'Reference 5':positions[:,4], 'Reference 6':positions[:,5],
    
    'M1 STA':motor1.w1[1:],
    'DM1 STA':motor1.w2[1:],
    'M2 STA':motor2.w1[1:],
    'DM2 STA':motor2.w2[1:],
    'M3 STA':motor3.w1[1:],
    'DM3 STA':motor3.w2[1:],
    'M4 STA':motor4.w1[1:],
    'DM4 STA':motor4.w2[1:],
    'M5 STA':motor5.w1[1:],
    'DM5 STA':motor5.w2[1:],
    'M6 STA':motor6.w1[1:],
    'DM6 STA':motor6.w2[1:],

    'LQR Value function':functional[:,0],
    'LQR Integral value function':np.cumsum(functional[:,0]),

    'Control 1':control[:,0],
    'Control 2':control[:,1],
    'Control 3':control[:,2],
    'Control 4':control[:,3],
    'Control 5':control[:,4],
    'Control 6':control[:,5]
}
    return dataAquired

def loadRLNN():
    # Load NNs
    path_actor = "C:\\Users\\guarn\\Dropbox\\Alejandro\\DoctoradoITESM\\Overleaf\\SGP control\\Stewart platform physical control\\actorRLSGP.onnx"
    model_actor = onnx.load(path_actor)
    onnx.checker.check_model(model_actor)
    actor_nn = ort.InferenceSession(path_actor)

    path_critic = "C:\\Users\\guarn\\Dropbox\\Alejandro\\DoctoradoITESM\\Overleaf\\SGP control\\Stewart platform physical control\\criticRLSGP1.onnx"
    model_critic = onnx.load(path_critic)
    onnx.checker.check_model(model_critic)
    critic_nn = ort.InferenceSession(path_critic)
    return actor_nn,critic_nn