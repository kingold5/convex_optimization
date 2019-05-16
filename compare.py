import numpy as np
import matplotlib.pyplot as plt
import settings

settings.init()

gpu_time = np.loadtxt(settings.Dir_PERFORMANCE+"/GPU_time.txt")
gpu_errors = np.loadtxt(settings.Dir_PERFORMANCE+"/GPU_errors.txt")
cpu_time = np.loadtxt(settings.Dir_PERFORMANCE+"/CPU_time.txt")
cpu_errors = np.loadtxt(settings.Dir_PERFORMANCE+"/CPU_errors.txt")

gpu_errors = np.log10(gpu_errors)
cpu_errors = np.log10(cpu_errors)

fig, ax = plt.subplots()
ax.plot(cpu_time, cpu_errors, label='CPU', color='blue')
ax.plot(gpu_time, gpu_errors, label='GPU', color='red')
legend = ax.legend(loc='upper right', fontsize='x-large')
plt.xlabel('time/s')
plt.ylabel('errors/log10')
plt.show()
