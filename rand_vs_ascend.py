# -*- coding: utf-8 -*-

import time
import numpy as np
import matplotlib.pyplot as plt
from gpu_calculation import GPU_Calculation
from parameters import parameters
from lasso import lasso, lasso_rand
from average import list_aver
import settings

settings.init()


def plt_fig(err_lasso, time_lasso, err_lasso_r, time_lasso_r):
    err_log_aver = np.log10(list_aver(err_lasso))
    time_aver = list_aver(time_lasso)

    err_log_aver_r = np.log10(list_aver(err_lasso_r))
    time_aver_r = list_aver(time_lasso_r)

    fig, ax = plt.subplots()
    ax.plot(time_aver, err_log_aver,
            label='ascend index', color='blue')
    ax.plot(time_aver_r, err_log_aver_r,
            label='random index', color='red')
    ax.legend(loc='upper right', fontsize='x-large')
    plt.xlabel('time/s')
    plt.ylabel('error/log10')
    plt.show()


# density of sparse vector
DENSITY = 0.4
# error bound
ERR_BOUND = 1e-4
# generating matrix and not save
READ_FLAG = False
SAVE_FLAG = False
INSTANCE = 50


for n_exp in np.arange(7, 8):
    N = 2 ** n_exp
    # was 1-16(0, 4)
    for k_plus in np.arange(3, 4):
        k_exp = n_exp + k_plus
        K = 2 ** k_exp
        # was np.arange(1+k_plus)
        for b_exp in np.arange(3, 4):
            BLOCK = 2 ** b_exp
            # lasso with ascend index
            time_comp = []
            err_lasso = []
            time_lasso = []

            # lasso with random index
            time_comp_r = []
            err_lasso_r = []
            time_lasso_r = []

            time_winit = 0
            ITER_MAX = 1000 * BLOCK

            for ii in range(INSTANCE):
                # make sure every random seed is unique
                time.sleep(1)
                (A, x_true, b, mu) = parameters(
                    N, K, DENSITY,
                    SAVE_FLAG, READ_FLAG, SILENCE=False)

                t_init = time.time()
                # set thread width
                GPU_Calculation.T_WIDTH_TRANS = 16
                GPU_Calculation.T_WIDTH = 16
                gpu_cal = GPU_Calculation(A, BLOCK)
                d_ATA = gpu_cal.diag_ATA
                time_winit += time.time() - t_init

                # let gpu warmup the first round
                lasso(gpu_cal, d_ATA, A, b, mu, BLOCK,
                      ERR_BOUND, ITER_MAX, SILENCE=False)
                time_comp.append(
                    lasso(gpu_cal, d_ATA, A, b, mu, BLOCK,
                          ERR_BOUND, ITER_MAX,
                          err_lasso, time_lasso,
                          SILENCE=False, DEBUG=False))

                time_comp_r.append(
                    lasso_rand(gpu_cal, d_ATA, A, b, mu, BLOCK,
                               ERR_BOUND, ITER_MAX,
                               err_lasso_r, time_lasso_r,
                               SILENCE=False, DEBUG=False))

            time_comp = np.asarray(time_comp)
            time_comp_r = np.asarray(time_comp_r)

            print('matrix size ({:-5}, {:-5})'.format(N, K)
                  + ', block size: ' + str(BLOCK)
                  + ', T_WIDTH=T_WIDTH_TRANS length: '
                  + str(GPU_Calculation.T_WIDTH)
                  + ', Ascend index computation time:'
                  + f'{np.mean(time_comp):.6f}' + ' s'
                  + ', standard deviation: '
                  + f'{np.std(time_comp):.6f}')
            print('Random index lasso computation time:',
                  np.mean(time_comp_r), 's.',
                  'standard deviation', np.std(time_comp_r))
            # print(time_winit / INSTANCE, 's.')
            plt_fig(err_lasso, time_lasso, err_lasso_r, time_lasso_r)
