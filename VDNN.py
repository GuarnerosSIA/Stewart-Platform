import numpy as np
import matplotlib.pyplot as plt
from controlClasses.algorithms import ValueDNN
from controlClasses.functions import DoPri45Step
from controlClasses.constants import*

t = 150000
dt = 0.005


dnnV = ValueDNN(QLQR1,RLQR1,BLQR1,ALQR1,PLQR1,alpha,beta,dt,w0,c)
delta = np.random.random((2,1))*10
weights = [dnnV.w0[0][0]]

steps = 50000
    
p11 = []
p21 = []
p12 = []
p22 = []


print(dnnV.gamma)
print(dnnV.phi1)
print(dnnV.phi2)
print(dnnV.phi3)

for i in range(steps-1):
    dnnV.pUpdate()
    P = dnnV.P[-1]
    Pdot = dnnV.pEquation(0,P)
    dnnV.wUpdate(delta)
    delta=delta*0.99
    p11.append(P[0][0])
    p21.append(P[1][0])
    p12.append(P[0][1])
    p22.append(P[1][1])
    weights.append(dnnV.w0[0][0])
    
print(P)

# p11 = [x[0,0] for x in dnnV.P]
# p21 = [x[1,0] for x in dnnV.P]
# p12 = [x[0,1] for x in dnnV.P]
# p22 = [x[1,1] for x in dnnV.P]

# plt.plot(p11)
# plt.plot(p21)
# plt.plot(p12)
# plt.plot(p22)

plt.plot(weights)

plt.show()


# print(dnnV.P[-1])