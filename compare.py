import numpy as np
import matplotlib.pyplot as plt
import settings

settings.init()

GPU_TIME = np.loadtxt(settings.HOME+"/Documents/python/GPU_time.txt")
GPU_ERRORS = np.loadtxt(settings.HOME+"/Documents/python/GPU_errors.txt")
CPU_TIME = np.loadtxt(settings.HOME+"/Documents/python/CPU_time.txt")
CPU_ERRORS = np.loadtxt(settings.HOME+"/Documents/python/CPU_errors.txt")

fig, ax = plt.subplots()
ax.plot(CPU_TIME, CPU_ERRORS, label='CPU', color='blue')
ax.plot(GPU_TIME, GPU_ERRORS, label='GPU', color='red')
legend = ax.legend(loc='upper right', fontsize='x-large')
plt.xlabel('time/s')
plt.ylabel('errors')
