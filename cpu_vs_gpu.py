# -*- coding: utf-8 -*-

import time
import numpy as np
import matplotlib.pyplot as plt
from gpu_calculation import GPU_Calculation
from cpu_calculation import A_bp_get, fun_diag_ATA
from parameters import parameters
from lasso import ClassLasso, ClassLassoR, ClassLassoCPU
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
INSTANCE = 10
ITER_MAX = 1000
P = 4

# initial time and error recording array
t_lasso = np.zeros((INSTANCE, ITER_MAX+1))
e_lasso = np.zeros((INSTANCE, ITER_MAX))
t_lasso_r = np.zeros_like(t_lasso)
e_lasso_r = np.zeros_like(e_lasso)
t_comp = np.zeros((INSTANCE))
t_comp_r = np.zeros_like(t_comp)
t_comp_cpu = np.zeros_like(t_comp)

for n_exp in np.arange(9, 10):
    N = 2 ** n_exp
    # was 1-16(0, 4)
    for k_plus in np.arange(3, 4):
        k_exp = n_exp + k_plus
        K = 2 ** k_exp
        # was np.arange(1+k_plus)
        for b_exp in np.arange(1, 2):
            BLOCK = 2 ** b_exp
            for t_width in np.arange(4, 5):
                # set thread width
                GPU_Calculation.T_WIDTH_TRANS = 2 ** t_width
                GPU_Calculation.T_WIDTH = 2 ** t_width

                # lasso computation time
                t_comp.fill(0)
                t_comp_r.fill(0)
                t_comp_cpu.fill(0)
                # reset time & error recording array
                t_lasso.fill(0)
                e_lasso.fill(0)
                t_lasso_r.fill(0)
                e_lasso_r.fill(0)
                time_winit = 0

                for i in range(INSTANCE):
                    # make sure every random seed is unique
                    time.sleep(1)
                    (A, x_true, b, mu) = parameters(
                        N, K, DENSITY,
                        SAVE_FLAG, READ_FLAG, SILENCE=True)

                    t_init = time.time()
                    gpu_cal = GPU_Calculation(A, BLOCK)
                    A_block_p = A_bp_get(A, BLOCK, P)
                    d_ATA = gpu_cal.diag_ATA
                    d_ATA_c = fun_diag_ATA(A_block_p)
                    print(d_ATA - d_ATA_c)
                    lasso = ClassLasso(gpu_cal, d_ATA, A, b, mu,
                                       BLOCK, ITER_MAX)
                    lasso_r = ClassLassoR(gpu_cal, d_ATA, A, b,
                                          mu, BLOCK, ITER_MAX)
                    lasso_cpu = ClassLassoCPU(A_block_p, d_ATA_c, A,
                                              b, mu, BLOCK, P, ITER_MAX)
                    time_winit += time.time() - t_init

                    # let gpu warmup the first round
                    # lasso(gpu_cal, d_ATA, A, b, mu, BLOCK,
                    #       ITER_MAX, ERR_BOUND, SILENCE=True)
                    t_comp[i] = lasso.run(ERR_BOUND,
                                          SILENCE=False,
                                          DEBUG=False)
                    t_comp_r[i] = lasso_r.run(ERR_BOUND,
                                              SILENCE=False)
                    t_comp_cpu[i] = lasso_cpu.run(ERR_BOUND,
                                                  SILENCE=False,
                                                  DEBUG=False)
                    print('')
                    # time_comp.append(
                    #     lasso(gpu_cal, d_ATA, A, b, mu, BLOCK,
                    #           ITER_MAX, ERR_BOUND,
                    #           err_iter=e_lasso[ii], time_iter=t_lasso[ii],
                    #           SILENCE=True, DEBUG=False))

                    # time_comp_r.append(
                    #     lasso_rand(gpu_cal, d_ATA, A, b, mu, BLOCK,
                    #                ITER_MAX, ERR_BOUND,
                    #                err_iter=e_lasso_r[ii],
                    #                time_iter=t_lasso_r[ii],
                    #                SILENCE=True, DEBUG=False))

                print('matrix size ({:-5}, {:-5})'.format(N, K)
                      + ', block: ' + str(BLOCK)
                      + ', T_WIDTH=T_WIDTH_TRANS: '
                      + str(GPU_Calculation.T_WIDTH)
                      + ', Ascend index GPU computation time: '
                      + f'{np.mean(t_comp):.6f}' + ' s'
                      + ', standard deviation: '
                      + f'{np.std(t_comp):.6f}')

                print('matrix size ({:-5}, {:-5})'.format(N, K)
                      + ', block: ' + str(BLOCK)
                      + ', T_WIDTH=T_WIDTH_TRANS: '
                      + str(GPU_Calculation.T_WIDTH)
                      + ', Random index GPU computation time: '
                      + f'{np.mean(t_comp_r):.6f}' + ' s'
                      + ', standard deviation: '
                      + f'{np.std(t_comp_r):.6f}')

                print('matrix size ({:-5}, {:-5})'.format(N, K)
                      + ', block: ' + str(BLOCK)
                      + ', Processor(s) used: ' + str(P)
                      + ', Ascend index CPU computation time:'
                      + f'{np.mean(t_comp_cpu):.6f}' + ' s'
                      + ', standard deviation: '
                      + f'{np.std(t_comp_cpu):.6f}')
                # print(time_winit / INSTANCE, 's.')
                # plt_fig(err_lasso, time_lasso, err_lasso_r, time_lasso_r)
