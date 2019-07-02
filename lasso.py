# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 19:05:53 2019

@author: Xingran Wang
"""

from multiprocessing import Pool
from itertools import product
import time
import numpy as np
import random
from pycuda import gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from skcuda import cublas
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel
from cpu_calculation import element_proj, soft_thresholding, error_crit,\
        fun_s12, fun_s22, fun_dd_p
import settings

settings.init()


# Pure CPU algorithm
class ClassLassoCPU:
    def __init__(self, A_block_p, d_ATA, A, b, mu, BLOCK, P, ITER_MAX):
        self.A_block_p = A_block_p
        self.d_ATA = d_ATA
        self.d_ATA_rec = [np.divide(1, self.d_ATA[i])
                          for i in range(BLOCK)]
        self.A = A
        self.A_SHAPE = A.shape
        self.b = b
        self.mu = mu
        self.BLOCK = BLOCK
        self.P = P
        self.ITER_MAX = ITER_MAX
        self.descript = 'CPU ascend index'

    def index_get(self, t):
        return t % self.BLOCK

    def debug(self, result_s13, x_block, Ax, b_k, t, m, r):
        if self.DEBUG:
            self.error = error_crit(result_s13, x_block, self.mu)
            opti_value2 = 0.5*np.sum(np.power(Ax[m]-b_k, 2)) +\
                self.mu*np.sum(np.abs(x_block))
            print('Loop {:-4} block {:-2} updated, '
                  'with Error {:.8f}, '
                  'optimum value {:4.6f}, '
                  'Stepsize {:.6f}'.format(
                      t, m, self.error, opti_value2, r))

    def fun_b_k(self, b_k, Ax, b, m):
        b_k[:] = b[:] 
        for i in range(self.BLOCK):
            if m != i:
                b_k -= Ax[i]

    def err_record(self, err_iter, result_s13, x_block, t):
        if self.ERR_RCD:
            if not self.DEBUG:
                self.error = error_crit(result_s13, x_block, self.mu)
            err_iter[t] = self.error

    def time_record(self, time_iter, t, start):
        if self.TIME_RCD:
            time_iter[t+1] = time.time() - start

    def rlt_display(self, SILENCE, t_elapsed, t):
        if not SILENCE:
            print('{:>20}, time used: {:.8f} s, '
                  'with {:-4} loops, and block number: {:-2}.'.format(
                      self.descript, t_elapsed, t+1, self.BLOCK))

    def run(self, ERR_BOUND=None, err_iter=None, time_iter=None,
            SILENCE=False, DEBUG=False):
        # initialize
        self.DEBUG = DEBUG
        if isinstance(ERR_BOUND, float):
            IS_BOUNDED = True
        else:
            IS_BOUNDED = False

        if isinstance(err_iter, np.ndarray):
            self.ERR_RCD = True
        else:
            self.ERR_RCD = False

        if isinstance(time_iter, np.ndarray):
            self.TIME_RCD = True
        else:
            self.TIME_RCD = False

        x = np.zeros((self.A_SHAPE[1], 1), dtype=np.float64)
        x_block = np.asarray(np.vsplit(x, self.BLOCK), dtype=np.float64)
        Ax = np.zeros((self.BLOCK, self.A_SHAPE[0], 1), dtype=np.float64)
        b_k = np.empty_like(self.b, dtype=np.float64)
        result_s13 = np.zeros_like(x_block[0], dtype=np.float64)
        result_s11 = np.zeros_like(self.b, dtype=np.float64)
        rx = np.zeros_like(x_block[0], dtype=np.float64)
        soft_t = np.zeros_like(rx, dtype=np.float64)
        Bx = np.zeros_like(soft_t, dtype=np.float64)
        descent_D = np.zeros_like(x_block[0], dtype=np.float64)
        result_s23 = np.zeros_like(self.b, dtype=np.float64)

        block_Cnt = 0

        # time_mul = 0
        # time_mul_t = 0

        start = time.time()
        if self.TIME_RCD:
            time_iter[0] = 0
        pool = Pool(processes=self.P)
        for t in range(self.ITER_MAX):
            # select mth block
            m = self.index_get(t)
            self.fun_b_k(b_k, Ax, self.b, m)
            result_s11 = Ax[m] - b_k
            # result_s11 = np.sum(Ax, axis=0) - self.b
            # result_s12 = A_p^T*(Ax-b)
            result_s12 = pool.starmap(fun_s12, product(self.A_block_p[m],
                                                       (result_s11,)))
            # result__s13 = (result_s12)[p=1...P]
            result_s13 = np.vstack(result_s12)

            # s14
            rx = np.multiply(self.d_ATA[m], x_block[m]) - result_s13
            soft_t = soft_thresholding(rx, self.mu)
            # s15
            Bx = np.multiply(self.d_ATA_rec[m], soft_t)
            # result_s21 = Bx_p - x_p
            descent_D = Bx-x_block[m]

            result_s21 = fun_dd_p(self.P, descent_D)
            # result_s22 = A_P(Bx_P - X_P)
            result_s22 = pool.starmap(fun_s22,
                                      zip(self.A_block_p[m], result_s21))
            # result_s23 = A(Bx-x)
            result_s23 = np.sum(result_s22, axis=0)

            # stepsize
            r_1 = np.transpose(result_s11) @ result_s23 +\
                self.mu*(np.linalg.norm(Bx, ord=1) -
                         np.linalg.norm(x_block[m], ord=1))
            r_2 = np.transpose(result_s23) @ result_s23
            if r_2 == 0.0:
                print('r_2 is ZERO, could not divide ZERO!')
            else:
                r = np.float64(element_proj(-r_1/r_2, 0, 1))

            self.debug(result_s13, x_block[m], Ax, b_k, t, m, r)
            self.err_record(err_iter, result_s13, x_block[m], t)

            if IS_BOUNDED:
                if not (self.DEBUG & self.ERR_RCD):
                    self.error = error_crit(result_s13, x_block[m], self.mu)
                if self.error < ERR_BOUND:
                    block_Cnt += 1
                if self.BLOCK - 1 == m:
                    if block_Cnt == self.BLOCK:
                        break
                    else:
                        block_Cnt = 0

            # x(t+1) = x(t)+r(Bx(t)-x(t))
            x_block[m] += r*descent_D
            # Ax(t+1)
            Ax[m] += r*result_s23
            opti_value2 = 0.5*np.sum(np.power(Ax[m]-b_k, 2)) +\
                self.mu*np.sum(np.abs(x_block[m]))
            print(opti_value2)

            self.time_record(time_iter, t, start)
            # print("matrix@vector:", time_mul,
            #       "s, matrix.T@vector:", time_mul_t)

        if self.TIME_RCD:
            t_elapsed = time_iter[t]
        else:
            t_elapsed = time.time() - start

        self.rlt_display(SILENCE, t_elapsed, t)
        x = np.vstack(x_block)

        return t_elapsed


# CPU & CUDA combined
class ClassLasso(ClassLassoCPU):
    def __init__(self, gpu_cal, d_ATA, A, b, mu, BLOCK, ITER_MAX):
        ClassLassoCPU.__init__(self, None, d_ATA, A, b, mu, BLOCK, None,
                               ITER_MAX)
        del self.A_block_p
        del self.P
        self.gpu_cal = gpu_cal
        self.descript = 'GPU ascend index'

    # matrix.T @ vector
    def _mtv(self, s13, m, s11):
        self.gpu_cal.mat_tMulVec_DiffSize(s13, m, s11)

    # matrix @ vector
    def _mv(self, s23, m, descent_D):
        self.gpu_cal.matMulVec_DiffSize(s23, m, descent_D)

    def time_record(self, time_iter, t, start_event, end_event):
        if self.TIME_RCD:
            end_event.record()
            end_event.synchronize()
            time_iter[t+1] = end_event.time_since(start_event) / 1e3

    def run(self, ERR_BOUND=None, err_iter=None, time_iter=None,
            SILENCE=False, DEBUG=False):
        # initialize
        self.DEBUG = DEBUG
        if isinstance(ERR_BOUND, float):
            IS_BOUNDED = True
        else:
            IS_BOUNDED = False

        if isinstance(err_iter, np.ndarray):
            self.ERR_RCD = True
        else:
            self.ERR_RCD = False

        if isinstance(time_iter, np.ndarray):
            self.TIME_RCD = True
        else:
            self.TIME_RCD = False

        # preallocation
        x = np.zeros((self.A_SHAPE[1], 1))
        x_block = np.asarray(np.vsplit(x, self.BLOCK))
        Ax = np.zeros((self.BLOCK, self.A_SHAPE[0], 1))
        b_k = np.zeros_like(self.b, dtype=np.float64)
        result_s13 = np.zeros_like(x_block[0])
        result_s11 = np.zeros_like(self.b, dtype=np.float64)
        rx = np.zeros_like(x_block[0], dtype=np.float64)
        soft_t = np.zeros_like(rx, dtype=np.float64)
        Bx = np.zeros_like(soft_t, dtype=np.float64)
        descent_D = np.zeros_like(x_block[0], dtype=np.float64)
        result_s23 = np.zeros_like(self.b)

        block_Cnt = 0

        start_fun = cuda.Event()
        end_fun = cuda.Event()
        start_event = cuda.Event()
        end_event = cuda.Event()

        time_f = 0
        start = time.time()
        start_event.record()
        if self.TIME_RCD:
            time_iter[0] = 0

        for t in range(self.ITER_MAX):
            # select mth block
            m = self.index_get(t)
            self.fun_b_k(b_k, Ax, self.b, m)
            result_s11 = Ax[m] - b_k
            # result_s11 = np.sum(Ax, axis=0) - self.b

            # result_s13 = gpu_cal.mat_tmulvec(m, result_s11)
            # start_fun.record()
            self._mtv(result_s13, m, result_s11)
            # end_fun.record()
            # end_fun.synchronize()
            # time_f += start_fun.time_till(end_fun)

            # s14
            rx = np.multiply(self.d_ATA[m], x_block[m]) - result_s13
            soft_t = soft_thresholding(rx, self.mu)
            # s15
            Bx = np.multiply(self.d_ATA_rec[m], soft_t)
            # result_s21 = Bx_p - x_p
            descent_D = Bx-x_block[m]

            # result_s23 = gpu_cal.matmulvec(m, descent_D)
            # result_s23 = gpu_cal.matMulVec_DiffSize(m, descent_D)
            self._mv(result_s23, m, descent_D)
            # stepsize
            r_1 = np.transpose(result_s11) @ result_s23 +\
                self.mu*(np.linalg.norm(Bx, ord=1) -
                         np.linalg.norm(x_block[m], ord=1))
            r_2 = np.transpose(result_s23) @ result_s23
            if r_2 == 0.0:
                print('r_2 is ZERO, could not divide ZERO!')
            else:
                r = np.float64(element_proj(-r_1/r_2, 0, 1))

            self.debug(result_s13, x_block[m], Ax, b_k, t, m, r)
            self.err_record(err_iter, result_s13, x_block[m], t)

            if IS_BOUNDED:
                if not (self.DEBUG & self.ERR_RCD):
                    self.error = error_crit(result_s13, x_block[m], self.mu)
                if self.error < ERR_BOUND:
                    block_Cnt += 1
                if self.BLOCK - 1 == m:
                    if block_Cnt == self.BLOCK:
                        break
                    else:
                        block_Cnt = 0

            # x(t+1) = x(t)+r(Bx(t)-x(t))
            x_block[m] += r*descent_D
            # Ax(t+1)
            Ax[m] += r*result_s23

            self.time_record(time_iter, t, start_event, end_event)
            # print("matrix@vector:", time_mul,
            #       "s, matrix.T@vector:", time_mul_t)

        end_event.record()
        end_event.synchronize()
        if self.TIME_RCD:
            t_elapsed = time_iter[t]
        else:
            t_elapsed = start_event.time_till(end_event) / 1e3

        self.rlt_display(SILENCE, t_elapsed, t)
        x = np.vstack(x_block)
        # if not SILENCE:
        #     print(str(time_f/1e3) + ' s.')

        return t_elapsed


# CPU & CUDA combined with random block index
class ClassLassoR(ClassLasso):
    def __init__(self, gpu_cal, d_ATA, A, b, mu, BLOCK, ITER_MAX):
        ClassLasso.__init__(self, gpu_cal, d_ATA, A,
                            b, mu, BLOCK, ITER_MAX)
        self.idx_shuffle = np.arange(self.BLOCK)
        self.descript = 'GPU random index'

    def index_get(self, t):
        if t % self.BLOCK == 0:
            random.shuffle(self.idx_shuffle)
        return self.idx_shuffle[t % self.BLOCK]


# CPU & cuBlas combined
class ClassLassoCB_v1(ClassLasso):
    def __init__(self, h, gpu_cal, d_ATA,
                 A, b, mu, BLOCK, ITER_MAX):
        ClassLasso.__init__(self, gpu_cal, d_ATA, A,
                            b, mu, BLOCK, ITER_MAX)
        self.descript = 'Cublas CPU combined'
        self.h = h

        self.idx_m = self.gpu_cal.MAT_HEIGHT
        self.idx_n = self.gpu_cal.MAT_WIDTH
        self.result_s13 = np.zeros(
            (self.idx_n, 1))
        self.result_s23 = np.zeros(
            (self.idx_m, 1))
        self.s13_gpu = gpuarray.zeros(
            (self.idx_n, 1), np.float64)
        self.s11_gpu = gpuarray.zeros(
            (self.idx_m, 1), np.float64)
        self.s23_gpu = gpuarray.zeros(
            (self.idx_m, 1), np.float64)
        self.d_d_gpu = gpuarray.zeros(
            (self.idx_n, 1), np.float64)

    # rewrite m @ v and m.T @ v
    def _mtv(self, result_s13, m, result_s11):
        self.s11_gpu.set(result_s11)
        self._zmvG(self.h, self.s13_gpu, 1, self.gpu_cal.A_b_gpu[m],
                   cublas._CUBLAS_OP['N'], self.s11_gpu)
        self.s13_gpu.get(result_s13)

    def _mv(self, result_s23, m, descent_D):
        self.d_d_gpu.set(descent_D)
        self._zmvG(self.h, self.s23_gpu, 1, self.gpu_cal.A_b_gpu[m],
                   cublas._CUBLAS_OP['T'], self.d_d_gpu)
        self.s23_gpu.get(result_s23)

    # z = a * m @ v
    def _zmvG(self, handle, z_gpu, alpha, m_gpu, trans, v_gpu):
        beta = 0
        n, m = m_gpu.shape
        cublas.cublasDgemv(handle, trans, m, n, alpha,
                           m_gpu.gpudata,
                           m, v_gpu.gpudata, 1, beta,
                           z_gpu.gpudata, 1)


# pure cuBlas
class ClassLassoCB_v2(ClassLasso):
    def __init__(self, h, gpu_cal, d_ATA,
                 A, b, mu, BLOCK, ITER_MAX):
        ClassLasso.__init__(self, gpu_cal, d_ATA, A,
                            b, mu, BLOCK, ITER_MAX)
        self.descript = 'Pure Cublas'
        self.h = h
        self.idx_m = self.gpu_cal.MAT_HEIGHT
        self.idx_n = self.gpu_cal.MAT_WIDTH

        self.x = np.empty((self.A_SHAPE[1], 1)).astype(np.float64)
        self.x_block = np.vsplit(self.x, self.BLOCK)
        self.Ax = np.empty(
            (self.BLOCK, self.A_SHAPE[0], 1)).astype(np.float64)
        self.x_gpu = gpuarray.to_gpu(self.x)
        self.x_block_gpu = [gpuarray.to_gpu(self.x_block[i])
                            for i in range(self.BLOCK)]
        self.Ax_gpu = gpuarray.to_gpu(self.Ax)
        self.b_gpu = gpuarray.to_gpu(self.b)

        '''
        self.result_s13 = np.zeros(
            (self.idx_n, 1))
        self.result_s23 = np.zeros(
            (self.idx_m, 1))
        self.s13_gpu = gpuarray.zeros(
            (self.idx_n, 1), np.float64)
        self.s11_gpu = gpuarray.zeros(
            (self.idx_m, 1), np.float64)
        self.s23_gpu = gpuarray.zeros(
            (self.idx_m, 1), np.float64)
        self.d_d_gpu = gpuarray.zeros(
            (self.idx_n, 1), np.float64)
        '''

        self.zsoft_t = ElementwiseKernel(
            'double *soft_t, double *tensor, double thres',
            '''
            soft_t[i] = copysign(1.0, tensor[i]) *\
                fmax(fabs(tensor[i]) - thres, 0.0);
            ''',
            'zsoft_t')

        self._err_array = ElementwiseKernel(
            'double *rlt, double *s13, double *x, double mu',
            'rlt[i] = s13[i] - fmax(fmin(s13[i]-x[i], mu), -mu);',
            '_err_array')

    def _mtv(self, handle, index, descent_D):
        alpha = 1
        beta = 0
        self.d_d_gpu.set(descent_D)
        trans = cublas._CUBLAS_OP['N']
        cublas.cublasDgemv(handle, trans, self.idx_m, self.idx_n, alpha,
                           self.gpu_cal.A_b_cw_gpu[index].gpudata,
                           self.idx_m, self.d_d_gpu.gpudata, 1,
                           beta, self.s23_gpu.gpudata, 1)

        return self.s23_gpu.get()

    # z = a * m @ v
    def _zmvG(self, handle, z_gpu, alpha, m_gpu, trans, v_gpu):
        beta = 0
        n, m = m_gpu.shape
        cublas.cublasDgemv(handle, trans, m, n, alpha,
                           m_gpu.gpudata,
                           m, v_gpu.gpudata, 1, beta,
                           z_gpu.gpudata, 1)

    # z = a * x + y
    def _zaxpy(self, handle, z_gpu, alpha, x_gpu, y_gpu):
        # copy y to z
        cublas.cublasDcopy(handle, y_gpu.size,
                           y_gpu.gpudata, 1, z_gpu.gpudata, 1)
        cublas.cublasDaxpy(handle, x_gpu.size, alpha,
                           x_gpu.gpudata, 1, z_gpu.gpudata, 1)

    # y = a * x + y
    def _axpy(self, handle, alpha, x_gpu, y_gpu):
        cublas.cublasDaxpy(handle, x_gpu.size, alpha,
                           x_gpu.gpudata, 1, y_gpu.gpudata, 1)

    def _l1norm(self, handle, x_gpu):
        return cublas.cublasDasum(handle, x_gpu.size,
                                  x_gpu.gpudata, 1)

    def _l2norm(self, handle, x_gpu):
        return cublas.cublasDnrm2(self.h, x_gpu.size, x_gpu.gpudata, 1)

    def fun_b_k(self, b_k_gpu, k):
        cublas.cublasDcopy(self.h, self.b_gpu.size,
                           self.b_gpu.gpudata, 1,
                           b_k_gpu.gpudata, 1)
        for i in range(len(self.Ax_gpu)):
            if i != k:
                cublas.cublasDaxpy(self.h, b_k_gpu.size,
                                   np.float64(-1),
                                   self.Ax_gpu[i].gpudata,
                                   1, b_k_gpu.gpudata, 1)

    def r1_get(self, handle, s11_gpu, s23_gpu, Bx_gpu, x_block_gpu):
        return cublas.cublasDdot(handle, s11_gpu.size, s11_gpu.gpudata,
                                 1, s23_gpu.gpudata, 1) +\
                self.mu*(self._l1norm(handle, Bx_gpu) -
                         self._l1norm(handle, x_block_gpu))

    def err_chk(self, handle, err_gpu, s13_gpu, x_block_gpu, mu):
        self._err_array(err_gpu, s13_gpu, x_block_gpu, mu)
        idx = cublas.cublasIdamax(handle, err_gpu.size, err_gpu.gpudata, 1)
        self.error = np.abs(err_gpu[idx].get())[0]

    def debug_gpu(self, handle, err_gpu, l2obj_gpu, b_k_gpu,
                  s13_gpu, x_block_gpu, mu, t, m, r):
        if self.DEBUG:
            self.err_chk(handle, err_gpu, s13_gpu, x_block_gpu, mu)
            self._zaxpy(handle, l2obj_gpu, -1, b_k_gpu, self.Ax_gpu[m])
            opti_value2 = 0.5*cublas.cublasDdot(handle, l2obj_gpu.size,
                                                l2obj_gpu.gpudata, 1,
                                                l2obj_gpu.gpudata, 1) +\
                mu*cublas.cublasDasum(handle, x_block_gpu.size,
                                      x_block_gpu.gpudata, 1)

            print('Loop {:-4} block {:-2} updated, '
                  'with Error {:.8f}, '
                  'optimum value {:4.6f}, '
                  'Stepsize {:.6f}'.format(
                      t, m, self.error, opti_value2, r))

    def err_record_gpu(self, handle, err_iter, err_gpu, s13_gpu, x_block_gpu, mu, t):
        if self.ERR_RCD:
            if not self.DEBUG:
                self.err_chk(handle, err_gpu, s13_gpu, x_block_gpu, mu)
            err_iter[t] = self.error

    def run(self, ERR_BOUND=None, err_iter=None, time_iter=None,
            SILENCE=False, DEBUG=False):
        # initialize
        self.DEBUG = DEBUG
        if isinstance(ERR_BOUND, float):
            IS_BOUNDED = True
        else:
            IS_BOUNDED = False

        if isinstance(err_iter, np.ndarray):
            self.ERR_RCD = True
        else:
            self.ERR_RCD = False

        if isinstance(time_iter, np.ndarray):
            self.TIME_RCD = True
        else:
            self.TIME_RCD = False

        # self.x.fill(0)
        for i in range(self.BLOCK):
            # self.x_block[i].fill(0)
            # self.Ax.fill(0)
            self.x_block_gpu[i].fill(0)
            self.Ax_gpu[i].fill(0)
            self.x_gpu.fill(0)

        b_k_gpu = gpuarray.empty_like(self.b_gpu)
        rx_gpu = gpuarray.empty_like(self.x_block_gpu[0])
        soft_t_gpu = gpuarray.empty_like(self.x_block_gpu[0])
        Bx_gpu = gpuarray.empty_like(self.x_block_gpu[0])
        s11_gpu = gpuarray.zeros(
            (self.idx_m, 1), np.float64)
        s13_gpu = gpuarray.zeros(
            (self.idx_n, 1), np.float64)
        s23_gpu = gpuarray.zeros(
            (self.idx_m, 1), np.float64)
        d_d_gpu = gpuarray.zeros(
            (self.idx_n, 1), np.float64)
        d_ATA_gpu = [gpuarray.to_gpu(self.d_ATA[i])
                     for i in range(self.BLOCK)]
        d_ATA_rec_gpu = [gpuarray.to_gpu(1 / self.d_ATA[i])
                         for i in range(self.BLOCK)]
        err_gpu = gpuarray.empty_like(self.x_block_gpu[0])
        l2obj_gpu = gpuarray.empty_like(self.b_gpu)

        block_Cnt = 0

        start_fun = cuda.Event()
        end_fun = cuda.Event()
        start_event = cuda.Event()
        end_event = cuda.Event()

        time_f = 0
        start = time.time()
        start_event.record()
        if self.TIME_RCD:
            time_iter[0] = 0

        cuda.start_profiler()
        for t in range(self.ITER_MAX):
            # select mth block
            m = self.index_get(t)
            '''
            result_s11 = np.sum(self.Ax, axis=0) - self.b
            self.s11_gpu.set(result_s11)
            self._zmvG(self.h, self.s13_gpu, 1,
                       self.gpu_cal.A_b_gpu[m], self.s11_gpu)
            self.s13_gpu.get(self.result_s13)
            rx = np.multiply(self.d_ATA[m], self.x_block[m]) -\
                self.result_s13
            soft_t = soft_thresholding(rx, self.mu)
            Bx = np.multiply(self.d_ATA_rec[m], soft_t)
            # result_s21 = Bx_p - x_p
            descent_D = Bx-self.x_block[m]
            self.d_d_gpu.set(descent_D)
            self._zmvG(self.h, self.s23_gpu, 1,
                       self.gpu_cal.A_b_cw_gpu[m], self.d_d_gpu)
            self.s23_gpu.get(self.result_s23)
            # result_s23 = self._mv(m, descent_D)
            r_1 = np.transpose(result_s11) @ self.result_s23 +\
                self.mu*(np.linalg.norm(Bx, ord=1) -
                         np.linalg.norm(self.x_block[m], ord=1))
            r_2 = np.transpose(self.result_s23) @ self.result_s23
            if r_2 == 0.0:
                print('r_2 is ZERO, could not divide ZERO!')
            else:
                r = np.float64(element_proj(-r_1/r_2, 0, 1))
            # x(t+1) = x(t)+r(Bx(t)-x(t))
            self.x_block[m] += r*descent_D
            # Ax(t+1)
            self.Ax[m] += r*self.result_s23
            self.debug(result_s13, self.x_block[m], self.x, t, m, r)
            self.err_record(err_iter, result_s13, self.x_block[m], t)
            if IS_BOUNDED:
                if not (self.DEBUG & self.ERR_RCD):
                    self.error = error_crit(
                        result_s13, self.x_block[m], self.mu)
                if self.error < ERR_BOUND:
                    block_Cnt += 1
                if self.BLOCK - 1 == m:
                    if block_Cnt == self.BLOCK:
                        break
                    else:
                        block_Cnt = 0
            '''

            # '''
            # begin pure gpu calculation
            self.fun_b_k(b_k_gpu, m)
            self._zaxpy(self.h, s11_gpu, -1, b_k_gpu, self.Ax_gpu[m])
            self._zmvG(self.h, s13_gpu, 1, self.gpu_cal.A_b_gpu[m],
                       cublas._CUBLAS_OP['N'], s11_gpu)
            # s14
            d_ATA_gpu[m]._elwise_multiply(
                self.x_block_gpu[m], rx_gpu)
            self._axpy(self.h, -1, s13_gpu, rx_gpu)
            self.zsoft_t(soft_t_gpu, rx_gpu, self.mu)
            # s15
            d_ATA_rec_gpu[m]._elwise_multiply(soft_t_gpu, Bx_gpu)
            self._zaxpy(self.h, d_d_gpu, -1, self.x_block_gpu[m], Bx_gpu)
            self._zmvG(self.h, s23_gpu, 1, self.gpu_cal.A_b_gpu[m],
                       cublas._CUBLAS_OP['T'], d_d_gpu)

            # stepsize
            # r_1g = self.r1_get(self.h, s11_gpu, s23_gpu,
            #                    Bx_gpu, self.x_block_gpu[m])
            # start_fun.record()
            temp_1 = cublas.cublasDdot(
                self.h, s11_gpu.size, s11_gpu.gpudata,
                1, s23_gpu.gpudata, 1)
            # end_fun.record()
            # end_fun.synchronize()
            # time_f += start_fun.time_till(end_fun)
            temp_2 = self.mu*(
                cublas.cublasDasum(self.h, Bx_gpu.size, Bx_gpu.gpudata, 1) -
                cublas.cublasDasum(self.h, self.x_block_gpu[m].size,
                                   self.x_block_gpu[m].gpudata, 1))
            r_1g = temp_1 + temp_2
            r_2g = np.square(self._l2norm(self.h, s23_gpu))
            if r_2g == 0.0:
                print('r_2 is ZERO, could not divide ZERO!')
            else:
                r_g = np.float64(element_proj(-r_1g/r_2g, 0, 1))

            self.debug_gpu(self.h, err_gpu, l2obj_gpu, b_k_gpu, s13_gpu,
                           self.x_block_gpu[m], self.mu, t, m, r_g)
            self.err_record_gpu(self.h, err_iter, err_gpu, s13_gpu,
                                self.Ax_gpu[m], self.mu, t)

            if IS_BOUNDED:
                if not (self.DEBUG & self.ERR_RCD):
                    self.err_chk(self.h, err_gpu, s13_gpu, self.x_block_gpu[m], self.mu)
                if self.error < ERR_BOUND:
                    block_Cnt += 1
                if self.BLOCK - 1 == m:
                    if block_Cnt == self.BLOCK:
                        break
                    else:
                        block_Cnt = 0

            self._axpy(self.h, r_g, d_d_gpu, self.x_block_gpu[m])
            self._axpy(self.h, r_g, s23_gpu, self.Ax_gpu[m])

            # print(np.allclose(self.x_block_gpu[m].get(),
            #                   self.x_block[m]))
            # '''
            self.time_record(time_iter, t, start_event, end_event)
            # print("matrix@vector:", time_mul,
            #       "s, matrix.T@vector:", time_mul_t)

        end_event.record()
        end_event.synchronize()
        cuda.stop_profiler()

        if self.TIME_RCD:
            t_elapsed = time_iter[t]
        else:
            t_elapsed = start_event.time_till(end_event) / 1e3

        self.rlt_display(SILENCE, t_elapsed, t)
        self.x = np.vstack(self.x_block)
        # if not SILENCE:
        #     print(str(time_f/1e3) + ' s.')

        return t_elapsed
