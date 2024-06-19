import numpy as np
import matplotlib.pyplot as plt
from controlClasses.algorithms import ValueDNN
from controlClasses.functions import DoPri45Step
from controlClasses.constants import*

t = 1
dt = 0.0001

print(QLQR1)
print(QLQR2)

dnnV = ValueDNN(QLQR2,RLQR2,BLQR2,ALQR2,PLQR2,alpha,beta,dt,w0,c)
delta = np.random.random((4,1))*10
weights = [dnnV.w0[0][0]]

steps = 1000
    
plin = np.zeros((nStates*nStates,steps))


print(dnnV.gamma)
print(dnnV.phi1)
print(dnnV.phi2)
print(dnnV.phi3)

for i in range(steps-1):
    dnnV.pUpdate()
    P = dnnV.P[-1]
    Pdot = dnnV.pEquation(0,P)
    # dnnV.wUpdate(delta)
    # delta=delta*0.99
    plin[:,i] = Pdot.flatten()
    # weights.append(dnnV.w0[0][0])
    


# p11 = [x[0,0] for x in dnnV.P]
# p21 = [x[1,0] for x in dnnV.P]
# p12 = [x[0,1] for x in dnnV.P]
# p22 = [x[1,1] for x in dnnV.P]

plt.plot(plin.T[:-1,:])


# plt.plot(weights)

plt.show()


# print(dnnV.P[-1])