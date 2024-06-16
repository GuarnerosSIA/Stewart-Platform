import numpy as np
import matplotlib.pyplot as plt
from controlClasses.algorithms import ValueDNN
from controlClasses.functions import DoPri45Step
from scipy.integrate import RK45,solve_ivp,DOP853

t = 150000
dt = 0.001

nStates = 2
nInputs = 1
nNeuronsV = 5

cSigmoid = np.random.random((nStates, nInputs))

def sigmoid(x,c):
    return 1/(1+np.exp(c.T@x))


ADNN = np.zeros((2,2))
ADNN[:1,1:] = np.eye(1)
ADNN[1:,:1] = np.eye(1)*2
ADNN[1:,1:] = -np.eye(1)*1

BDNN = np.zeros((2,1))
BDNN[1:,:] = np.eye(1)

P0 =  np.zeros((2,2))
P0[0,0] = 0
P0[0,1] = 0
P0[1,0] = 0
P0[1,1] = 0


QDNN = np.zeros((2,2))
QDNN[:1,:1] = np.eye(1)*2
QDNN[1:,1:] = np.eye(1)*1

RDNN = np.eye(1)*0.5

alpha = 0.9
beta =  0.9

w0 = np.random.random((nNeuronsV,1))*10
c = np.random.random((nNeuronsV,nStates)).T*0.01

dnnV = ValueDNN(QDNN,RDNN,BDNN,ADNN,P0,alpha,beta,dt,w0,c)
delta = np.random.random((2,1))
print(dnnV.w0[0][0])
weights = [dnnV.w0[0][0]]

steps = 1000
t = np.linspace(0,15,steps)
Pint = []
time = []
P0vec = P0.flatten()
def riccati(t,p):
    # p = p.reshape((2,2))
    pa = p@ADNN
    ap = ADNN.T@p
    pb = p@BDNN
    q = QDNN
    rInv = np.linalg.inv(RDNN)
    bp = BDNN.T@p
    return -1*(-pa-ap-q + pb@rInv@bp)
    

p11 = []
p21 = []
p12 = []
p22 = []


for i in range(steps-1):
    h = t[i+1]-t[i]
    v4, v5 = DoPri45Step(riccati,t[i],P0,h)
    newP = P0 + h*v5
    
    p11.append(newP[0][0])
    p21.append(newP[1][0])
    p12.append(newP[0][1])
    p22.append(newP[1][1])
    P0 = newP

    # dnnV.pUpdate()
    # print(dnnV.valueFunction(delta))
    # dnnV.wUpdate(delta)
    # delta = delta*0.95
    # weights.append(dnnV.w0[0][0])
    
print(dnnV.w0[0][0])

# p11 = [x[0,0] for x in dnnV.P]
# p21 = [x[1,0] for x in dnnV.P]
# p12 = [x[0,1] for x in dnnV.P]
# p22 = [x[1,1] for x in dnnV.P]

plt.plot(t[:-1],p11)
plt.plot(t[:-1],p21)
plt.plot(t[:-1],p12)
plt.plot(t[:-1],p22)


plt.show()


print(dnnV.P[-1])