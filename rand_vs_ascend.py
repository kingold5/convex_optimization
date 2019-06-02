# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 19:05:53 2019

@author: Xingran Wang
"""

import time
import numpy as np
from gpu_calculation import GPU_Calculation
from parameters import parameters
from lasso import lasso
from lasso_rand import lasso_rand
import settings

settings.init()

# number of blocks
BLOCK = 4
# col of matrix A
K = 128
# row of matrix A
N = 32
# density of sparse vector
DENSITY = 0.4
# error bound
ERR_BOUND = 1e-4
# maximum number of iterations
ITER_MAX = 1000*BLOCK

# lasso with ascend index
time_comp = []
err_lasso = []
time_lasso = []

# lasso with random index
time_comp_r = []
err_lasso_r = []
time_lasso_r = []

INSTANCE = 50
REP = 1
time_winit = 0

for ii in range(INSTANCE):
    READ_FLAG = False
    SAVE_FLAG = False

    # make sure every random seed is unique
    time.sleep(1)
    (A, x_true, b, mu) = parameters(N, K, DENSITY, SAVE_FLAG, READ_FLAG)

    t_init = time.time()
    gpu_cal = GPU_Calculation(A, BLOCK)
    d_ATA = gpu_cal.diag_ATA
    time_winit += time.time() - t_init

    # let gpu warmup the first round
    lasso(gpu_cal, d_ATA, A, b, mu, BLOCK,
          ERR_BOUND, ITER_MAX)
    time_comp.append(
        lasso(gpu_cal, d_ATA, A, b, mu, BLOCK,
              ERR_BOUND, ITER_MAX, err_lasso, time_lasso))

    # let gpu warmup the first round
    # lasso_rand(gpu_cal, d_ATA, A, b, mu,
    #            BLOCK, ERR_BOUND, ITER_MAX)
    time_comp_r.append(
        lasso_rand(gpu_cal, d_ATA, A, b, mu, BLOCK,
                   ERR_BOUND, ITER_MAX, err_lasso_r, time_lasso_r))

time_comp = np.asarray(time_comp)
time_comp_r = np.asarray(time_comp_r)
print('Ascend index lasso computation time:',
      np.mean(time_comp), 's.',
      'standard deviation', np.std(time_comp))
print('Random index lasso computation time:',
      np.mean(time_comp_r), 's.',
      'standard deviation', np.std(time_comp_r))
print(time_winit / INSTANCE, 's.')
