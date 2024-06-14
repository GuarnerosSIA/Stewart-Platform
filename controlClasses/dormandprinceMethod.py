from scipy.integrate import RK45,solve_ivp
import numpy as np
import matplotlib.pyplot as plt

def sin_func(t,x):
    return -2*x*delta

def sin_func_args(t,x,delta):
    return -2*x*delta



t = np.linspace(0,np.pi,101)
x0 = [100]
y = []
yb = []
time = []
timeb = []
delta=-1
sol = RK45(sin_func,t[0],x0,t_bound=5,max_step=0.05)


for i in range(100):
    sol_args = solve_ivp(sin_func_args,[t[i],t[i+1]],x0,args=(delta,), method='DOP853')
    x0 = [sol_args.y[0][-1]]
    yb.append(x0[0])
    timeb.append(sol_args.t[-1])
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
plt.show()