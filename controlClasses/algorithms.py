import numpy as np

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

class STA:
    """
    Class for the supertwisting algorithm in continuous time
    """
    def __init__(self,tau,l1, l2, w1=0.0, w2=0.0):
        self.tau = tau
        self.w1 = [w1]
        self.w2 = [w2]
        self.l1 = l1
        self.l2 = l2
    
    def derivative(self,variable):
        error = self.w1[-1]-variable
        w1_aux = self.tau*(self.w2[-1] - self.l1*(np.sqrt(np.abs(error)))*self.sign(error)) + self.w1[-1]
        w2_aux = self.tau*(-self.l2*self.sign(error)) + self.w2[-1]
        self.w1.append(w1_aux)
        self.w2.append(w2_aux)
        
        return w2_aux
    
    def sign(self, value):
        if value < 0:
            out = -1
        else:
            out = 1
        return out
    
class DSTA(STA):
    """
    Class for supertwisting in discrete time
    """
    def __init__(self, tau, l1, l2, rho1, rho2, w1=0, w2=0):
        super().__init__(tau, l1, l2, w1, w2)
        self.rho1 = rho1
        self.rho2 = rho2

    def derivative(self,variable):
        error = self.w1[-1]-variable
        w1_aux = self.tau*(self.w2[-1] - self.l1*(np.sqrt(np.abs(error)))*np.sign(error)) + self.rho1*self.w1[-1]
        w2_aux = self.tau*(-self.l2*np.sign(error)) + self.rho2*self.w2[-1]
        self.w1.append(w1_aux)
        self.w2.append(w2_aux)
        
        return w2_aux
    
    
class LQR():
    """
    Class for implementing a PD contorller
    """
    def __init__(self,Q,R,B,A,P0,dt,epsilon):
        self.Q = Q
        self.R = R
        self.B = B
        self.A = A
        self.P = P0
        self.epsilon = epsilon
        self.dt = dt
        self.K = [-np.linalg.inv(self.R)@self.B.T@self.P]

    def gainsComputation(self):
        delta = 1
        pAnt =  self.P
        t = 0
        h = self.dt
        while delta > self.epsilon:
            v4, v5 = DoPri45Step(self.rDE,t,pAnt,h)
            newP = pAnt + h*v5
            delta = abs(np.linalg.norm(pAnt-newP))
            pAnt = newP
            self.K.append(-np.linalg.inv(self.R)@self.B.T@pAnt)
            t += h
        self.P = newP

    def opControl(self,delta):
        return self.K[-1]@delta
    
    def rDE(self,t,p):
        pa = p@self.A
        ap = self.A.T@p
        pb = p@self.B
        q = self.Q
        rInv = np.linalg.inv(self.R)
        bp = self.B.T@p
        return -1*(-pa-ap-q + pb@rInv@bp)
            

class ValueDNN():
    """
    Esta clase permite generar una red neuronal que aproxime la función Valor de una red 
    """
    def __init__(self,Q,R,B,A,P0,alpha,beta,dt,w0,c):
        """
        Valores iniciales de la ecuación diferencial de Riccati
        """
        self.Q = Q
        self.R = R
        self.B = B
        self.A = A
        self.gamma = 0.5*((alpha**2)+(beta**2))
        self.phi1 = self.gamma*np.eye(1) + R
        self.phi2 = 0.5*self.gamma*np.eye(2) + Q
        self.phi3 = 0.5*np.eye(2) -  0.25*B@np.linalg.inv(self.phi1)@B.T
        self.P = [P0]
        self.dt = dt
        self.t = 0
        self.w0 = w0
        self.nNeurons = w0.shape[0]
        self.c = c

    def pUpdate(self):
        p = self.P[-1]
        v4, v5 = DoPri45Step(self.pEquation,self.t,p,self.dt)
        newP = p + self.dt*v5
        self.P.append(newP)

    def pEquation(self,t,x):
        p = x
        pa = p@self.A
        ap = self.A.T@p
        pPhip = 4*p@self.phi3@p
        return 1*(-pa-ap-pPhip-self.phi2)

    def valueFunction(self, x):
        value = 0
        for i in range(self.nNeurons):
            sW = self.w0[i]**2
            sSig = self.sigmoid(self.c.T[i], x)**2
            value += sW*sSig
        Pdelta = x.T@self.P[-1]@x
        return value + Pdelta
    
    def nablaV(self,delta):
        value = 0
        for i in range(self.nNeurons):
            sW = self.w0[i]**2
            sDSig = self.dsigmoid(self.c.T[i],delta)
            sSig = self.sigmoid(self.c.T[i],delta)
            value += sW*sSig*sDSig*self.c.T[i]
        
        aux = 2*self.P[-1]@delta
        value = value.reshape((2,1))
        
        return value + aux
    
    def wUpdate(self,delta):
        suma = 0
        sW = np.square(self.w0)
        w0 = self.w0
        for i in range(self.nNeurons):
            sDSig = self.dsigmoid(self.c.T[i],delta)
            sSig = self.sigmoid(self.c.T[i],delta)
            suma += sW[i]*sSig*sDSig*self.c.T[i]
        suma = suma.reshape((-1,1))
        common = self.A@delta+self.phi3@(suma+4*self.P[-1]@delta)

        for i in range(self.nNeurons):
            sDSig = self.dsigmoid(self.c.T[i],delta)
            sSig = self.sigmoid(self.c.T[i],delta)
            frac = w0[i]*sSig*sDSig*self.c.T[i]/(2*(sSig**2))
            frac = frac.reshape((1,-1))
            self.w0[i] = -frac@common*self.dt + self.w0[i]
        
    
    def sigmoid(self, c, x):
        return 1/(1+np.exp(c.T@x))
    
    def dsigmoid(self, c, x):
        return self.sigmoid(c,x)*(1-self.sigmoid(c,x))
        

