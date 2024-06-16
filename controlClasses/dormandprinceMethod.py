from scipy.integrate import RK45,solve_ivp
import numpy as np
import matplotlib.pyplot as plt

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

def sin_func(t,x):
    return -2*x*delta

def sin_func_args(t,x,delta):
    return -2*x*delta



t = np.linspace(0,np.pi,101)
x0 = [100]
x00 = 100
y = []
yb = []
yManuak = []
time = []
timeb = []
timeManual = []
delta=-1
sol = RK45(sin_func,t[0],x0,t_bound=5,max_step=0.05)


for i in range(100):
    sol_args = solve_ivp(sin_func_args,[t[i],t[i+1]],x0,args=(delta,), method='DOP853')
    x0 = [sol_args.y[0][-1]]
    yb.append(x0[0])
    timeb.append(sol_args.t[-1])
    h = t[i+1]-t[i]
    v4, v5 = DoPri45Step(sin_func,t[i],x00,h)
    yManuak.append(x00 + h*v5)
    x00 = yManuak[-1]

    # print(x0)
    y.append(sol.y[0])
    time.append(sol.t)
    if(i>20):
        delta = 1
    else:
        delta = -1
    sol.step()
    
plt.plot(time,y)
new = [100*np.exp(-2*x) for x in time]
plt.plot(time,new)
plt.plot(timeb, yb)
plt.plot(t[:-1],yManuak)
plt.show()