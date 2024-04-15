import numpy as np

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
    def __init__(self,Q,R,B,A):
        self.Q = Q
        self.R = R
        self.B = B
        self.A = A
