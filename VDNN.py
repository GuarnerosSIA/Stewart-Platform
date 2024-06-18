import numpy as np
import matplotlib.pyplot as plt
from controlClasses.algorithms import ValueDNN
from controlClasses.functions import DoPri45Step

t = 150000
dt = 0.005

nStates = 2
nInputs = 1
nNeuronsV = 5

cSigmoid = np.random.random((nStates, nInputs))

def sigmoid(x,c):
    return 1/(1+np.exp(c.T@x))


ADNN = np.zeros((2,2))
ADNN[:1,1:] = np.eye(1)
ADNN[1:,:1] = -np.eye(1)*100
ADNN[1:,1:] = -np.eye(1)*50

BDNN = np.zeros((2,1))
BDNN[1:,:] = np.eye(1)

P0 =  np.zeros((2,2))
P0[0,0] = 5
P0[0,1] = 0
P0[1,0] = 0
P0[1,1] = 5


QDNN = np.zeros((2,2))
QDNN[:1,:1] = np.eye(1)*0.05
QDNN[1:,1:] = np.eye(1)*0.05

RDNN = np.eye(1)*0.001

alpha = 0.5
beta =  0.5

w0 = np.random.random((nNeuronsV,1))*10
c = np.random.random((nNeuronsV,nStates)).T*0.01

dnnV = ValueDNN(QDNN,RDNN,BDNN,ADNN,P0,alpha,beta,dt,w0,c)
delta = np.random.random((2,1))*10
weights = [dnnV.w0[0][0]]

steps = 50000
    
p11 = []
p21 = []
p12 = []
p22 = []

print(dnnV.pEquation(0,P0))
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