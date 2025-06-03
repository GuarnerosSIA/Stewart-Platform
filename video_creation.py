import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import rcParams


# Data creation
rcParams.update({'font.family': 'Times New Roman',
                 'font.size':33,
                 'text.usetex':True})

onyx = (0.2235, 0.2431, 0.2549)
cinnabar = (0.9137, 0.3098, 0.2157)
steelblue = (0.2471, 0.5333, 0.7725)


time = 10


# Leer el CSV
df = pd.read_csv('.\\data\\platformPD_boat.csv')
df_B = pd.read_csv('.\\data\\platformLQR_boat.csv')
data_size = df['System 1'].shape[0]
time = np.linspace(0, time, data_size)
valores_pd_1 = df['System 1']
valores_nn_1 = df_B['System 1']
reference_1 = df['Reference 1']


fig, ax = plt.subplots(figsize=(10, 5))

line_ref, = ax.plot([], [], label = 'Reference',linewidth=10, color = onyx)
line_pd, = ax.plot([], [], label = 'LQR+PD', linewidth=7, color = cinnabar, linestyle='-.')
line_nn, = ax.plot([], [], label = 'V-DNN', linewidth=7, color = steelblue, linestyle=':')


ax.set_xlim(2.5, 10)
ax.set_ylim(2.5, 8)
ax.set_xlabel(r'Time (s)', fontsize=20)
ax.set_ylabel(r'$x_1$', fontsize=20)
ax.legend(fontsize=20)
ax.grid(True)

def update(frame):
    line_pd.set_data(time[:frame], valores_nn_1[:frame])
    line_nn.set_data(time[:frame], valores_pd_1[:frame])
    line_ref.set_data(time[:frame], reference_1[:frame])
    
    return line_pd, line_nn, line_ref

ani = FuncAnimation(fig, update, frames=len(time), blit=True, interval=10)
ani.save('.\\videos\\boat.gif', writer='pillow', fps=30)
