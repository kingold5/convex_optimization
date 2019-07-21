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
        fun_s12, fun_s22
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
        self.descript = 'Pure CPU'

        # preallocate parameters
        self.x = np.zeros((self.A_SHAPE[1], 1), dtype=np.float64)
        self.x_block = np.asarray(
            np.vsplit(self.x, self.BLOCK), dtype=np.float64)
        self.Ax = np.zeros((self.BLOCK, self.A_SHAPE[0], 1), dtype=np.float64)
        self.b_k = np.empty_like(self.b, dtype=np.float64)
        self.result_s13 = np.zeros_like(self.x_block[0], dtype=np.float64)
        self.result_s11 = np.zeros_like(self.b, dtype=np.float64)
        self.rx = np.zeros_like(self.x_block[0], dtype=np.float64)
        self.soft_t = np.zeros_like(self.rx, dtype=np.float64)
        self.Bx = np.zeros_like(self.soft_t, dtype=np.float64)
        self.descent_D = np.zeros_like(self.x_block[0], dtype=np.float64)
        self.result_s21 = np.zeros_like(self.x_block[0], dtype=np.float64)
        self.result_s23 = np.zeros_like(self.b, dtype=np.float64)

    def index_get(self, t):
        return t % self.BLOCK

    def debug(self, DEBUG, pool, t, m, r):
        if DEBUG:
            self.result_s11 = self.Ax[m] - self.b_k
            result_s12 = pool.starmap(
                fun_s12, product(self.A_block_p[m], (self.result_s11,)))
            np.concatenate(result_s12, axis=0, out=self.result_s13)
            self.error = error_crit(
                self.result_s13, self.x_block[m], self.mu)
            # calculate objective function w.r.t. x
            # opti_value2 = 0.5*np.sum(np.square(self.A@x-self.b)) +\
            #     self.mu*np.sum(np.abs(x))
            # calculate objective function w.r.t. x_block
            opti_value2 = 0.5*np.sum(np.square(self.result_s11)) +\
                self.mu*np.sum(np.abs(self.x_block[m]))
            print('Loop {:-4} block {:-2} updated, '
                  'with Error {:.8f}, '
                  'optimum value {:4.6f}, '
                  'Stepsize {:.6f}'.format(
                      t, m, self.error, opti_value2, r))

    def fun_b_k(self, m):
        self.b_k[:] = self.b[:]
        for i in range(self.BLOCK):
            if m != i:
                self.b_k -= self.Ax[i]

    def err_record(self, err_iter, m, t):
        if self.ERR_RCD:
            if not self.DEBUG:
                self.error = error_crit(
                    self.result_s13, self.x_block[m], self.mu)
            err_iter[t] = self.error

    def bnd_chk(self, m, ERR_BOUND):
        # bnd_flag value
        # 0 break ------ all blocks satisfied criteria
        # 1 continue --- current block satisfies criteria
        # 2 go next ---- current block do not satisfies criteria
        # 4 err_chked -- do not perform bound check again
        if self.bnd_flag == 4:
            self.bnd_flag = 2
        else:
            self.bnd_flag = 2
            if self.IS_BOUNDED:
                if not self.ERR_RCD:
                    self.error = error_crit(
                        self.result_s13, self.x_block[m], self.mu)
                if self.error < ERR_BOUND:
                    if self.BLOCK - 1 != m:
                        self.block_Cnt += 1
                        self.bnd_flag = 1
                    elif self.block_Cnt != self.BLOCK - 1:
                        self.block_Cnt = 0
                        self.bnd_flag = 1
                    else:
                        self.bnd_flag = 0

    def time_record(self, time_iter, t, start):
        if self.TIME_RCD:
            time_iter[t+1] = time.time() - start

    def init_flags(self, ERR_BOUND, err_iter, time_iter):
        if isinstance(ERR_BOUND, float):
            self.IS_BOUNDED = True
        else:
            self.IS_BOUNDED = False

        if isinstance(err_iter, np.ndarray):
            self.ERR_RCD = True
        else:
            self.ERR_RCD = False

        if isinstance(time_iter, np.ndarray):
            self.TIME_RCD = True
        else:
            self.TIME_RCD = False

    def rlt_display(self, SILENCE, t_elapsed, t):
        if not SILENCE:
            print('{:>20}, time used: {:.8f} s, '
                  'with {:-4} loops, and block number: {:-2}.'.format(
                      self.descript, t_elapsed, t+1, self.BLOCK))

    def run(self, ERR_BOUND=None, err_iter=None, time_iter=None,
            SILENCE=False, DEBUG=False):
        self.init_flags(ERR_BOUND, err_iter, time_iter)
        # reset parameters as 0
        self.x.fill(0)
        self.x_block.fill(0)
        self.Ax.fill(0)

        self.block_Cnt = 0
        self.bnd_flag = -1

        start = time.time()
        if self.TIME_RCD:
            time_iter[0] = 0
        pool = Pool(processes=self.P)
        for t in range(self.ITER_MAX):
            # select mth block
            m = self.index_get(t)
            self.fun_b_k(m)
            self.result_s11 = self.Ax[m] - self.b_k
            # result_s11 = np.sum(Ax, axis=0) - self.b
            # result_s12 = A_p^T*(Ax-b)
            result_s12 = pool.starmap(
                fun_s12, product(self.A_block_p[m], (self.result_s11,)))
            # result__s13 = (result_s12)[p=1...P]
            np.concatenate(result_s12, axis=0, out=self.result_s13)

            # s14
            self.rx = np.multiply(self.d_ATA[m], self.x_block[m]) -\
                self.result_s13
            self.soft_t = soft_thresholding(self.rx, self.mu)
            # s15
            np.multiply(self.d_ATA_rec[m], self.soft_t, out=self.Bx)
            # check if Bx is near x
            if np.allclose(self.Bx, self.x_block[m], atol=1e-04):
                if not self.IS_BOUNDED:
                    continue
                else:
                    self.bnd_chk(m, ERR_BOUND)
                    if self.bnd_flag == 0:
                        break
                    elif self.bnd_flag == 1:
                        continue
                    else:
                        self.bnd_flag = 4

            # result_s21 = Bx_p - x_p
            self.descent_D = self.Bx-self.x_block[m]

            self.result_s21 = self.descent_D.reshape((self.P, -1, 1))
            # result_s22 = A_P(Bx_P - X_P)
            result_s22 = pool.starmap(
                fun_s22, zip(self.A_block_p[m], self.result_s21))
            # result_s23 = A(Bx-x)
            np.sum(result_s22, axis=0, out=self.result_s23)

            # stepsize
            r_1 = np.transpose(self.result_s11) @ self.result_s23 +\
                self.mu*(np.sum(np.abs(self.Bx) -
                         np.sum(np.abs(self.x_block[m]))
            r_2 = np.transpose(self.result_s23) @ self.result_s23
            if r_2 == 0.0:
                print('r_2 is ZERO, could not divide ZERO!')
            else:
                r = np.float64(element_proj(-r_1/r_2, 0, 1))

            self.err_record(err_iter, m, t)
            self.bnd_chk(m, ERR_BOUND)
            if self.bnd_flag == 0:
                break
            elif self.bnd_flag == 1:
                continue

            # x(t+1) = x(t)+r(Bx(t)-x(t))
            self.x_block[m] += r*self.descent_D
            # Ax(t+1)
            self.Ax[m] += r*self.result_s23
            self.debug(DEBUG, pool, t, m, r)
            self.time_record(time_iter, t, start)
            # print("matrix@vector:", time_mul,
            #       "s, matrix.T@vector:", time_mul_t)

        if self.TIME_RCD:
            t_elapsed = time_iter[t]
        else:
            t_elapsed = time.time() - start

        self.rlt_display(SILENCE, t_elapsed, t)
        self.x = np.vstack(self.x_block)

        return t_elapsed


class ClassLassoCPUEEC(ClassLassoCPU):
    def run(self, ERR_BOUND=None, err_iter=None, time_iter=None,
            SILENCE=False, DEBUG=False):
        self.init_flags(ERR_BOUND, err_iter, time_iter)
        # reset parameters as 0
        self.x.fill(0)
        self.x_block.fill(0)
        self.Ax.fill(0)

        self.block_Cnt = 0
        self.bnd_flag = -1

        start = time.time()
        if self.TIME_RCD:
            time_iter[0] = 0
        pool = Pool(processes=self.P)
        for t in range(self.ITER_MAX):
            # select mth block
            m = self.index_get(t)
            self.fun_b_k(m)
            self.result_s11 = self.Ax[m] - self.b_k
            # result_s11 = np.sum(Ax, axis=0) - self.b
            # result_s12 = A_p^T*(Ax-b)
            result_s12 = pool.starmap(
                fun_s12, product(self.A_block_p[m], (self.result_s11,)))
            # result__s13 = (result_s12)[p=1...P]
            np.concatenate(result_s12, axis=0, out=self.result_s13)
            # self.err_record(err_iter, m, t)
            self.bnd_chk(m, ERR_BOUND)
            if self.bnd_flag == 0:
                break
            elif self.bnd_flag == 1:
                continue

            # s14
            self.rx = np.multiply(self.d_ATA[m], self.x_block[m]) -\
                self.result_s13
            self.soft_t = soft_thresholding(self.rx, self.mu)
            # s15
            np.multiply(self.d_ATA_rec[m], self.soft_t, out=self.Bx)
            if not self.IS_BOUNDED:
                if np.allclose(self.Bx, self.x_block[m], atol=1e-04):
                    continue

            # result_s21 = Bx_p - x_p
            self.descent_D = self.Bx-self.x_block[m]
            self.result_s21 = self.descent_D.reshape((self.P, -1, 1))
            # result_s22 = A_P(Bx_P - X_P)
            result_s22 = pool.starmap(
                fun_s22, zip(self.A_block_p[m], self.result_s21))
            # result_s23 = A(Bx-x)
            np.sum(result_s22, axis=0, out=self.result_s23)

            # stepsize
            r_1 = np.transpose(self.result_s11) @ self.result_s23 +\
                self.mu*(np.sum(np.abs(self.Bx) -
                         np.sum(np.abs(self.x_block[m]))
            r_2 = np.transpose(self.result_s23) @ self.result_s23
            if r_2 == 0.0:
                print('r_2 is ZERO, could not divide ZERO!')
            else:
                r = np.float64(element_proj(-r_1/r_2, 0, 1))

            # x(t+1) = x(t)+r(Bx(t)-x(t))
            self.x_block[m] += r*self.descent_D
            # Ax(t+1)
            self.Ax[m] += r*self.result_s23
            self.debug(DEBUG, pool, t, m, r)
            self.time_record(time_iter, t, start)
            # print("matrix@vector:", time_mul,
            #       "s, matrix.T@vector:", time_mul_t)

        if self.TIME_RCD:
            t_elapsed = time_iter[t]
        else:
            t_elapsed = time.time() - start

        self.rlt_display(SILENCE, t_elapsed, t)
        self.x = np.vstack(self.x_block)

        return t_elapsed


# CPU & CUDA combined
class ClassLasso(ClassLassoCPU):
    def __init__(self, gpu_cal, d_ATA, A, b, mu, BLOCK, ITER_MAX):
        ClassLassoCPU.__init__(self, None, d_ATA, A, b, mu, BLOCK, None,
                               ITER_MAX)
        del self.A_block_p
        del self.P
        self.gpu_cal = gpu_cal
        self.descript = 'CUDA & CPU combined'

    # matrix.T @ vector
    def _mtv(self, m):
        self.gpu_cal.mat_tMulVec_DiffSize(self.result_s13, m, self.result_s11)

    # matrix @ vector
    def _mv(self, m):
        self.gpu_cal.matMulVec_DiffSize(self.result_s23, m, self.descent_D)

    def time_record(self, time_iter, t, start_event, end_event):
        if self.TIME_RCD:
            end_event.record()
            end_event.synchronize()
            time_iter[t+1] = end_event.time_since(start_event) / 1e3

    def debug(self, DEBUG, t, m, r):
        if DEBUG:
            self.result_s11 = self.Ax[m] - self.b_k
            self._mtv(m)
            self.error = error_crit(self.result_s13, self.x_block, self.mu)
            # calculate objective function w.r.t. x
            # opti_value2 = 0.5*np.sum(np.square(self.A@x-self.b)) +\
            #     self.mu*np.sum(np.abs(x))
            # calculate objective function w.r.t. x_block
            opti_value2 = 0.5*np.sum(np.square(self.result_s11)) +\
                self.mu*np.sum(np.abs(self.x_block[m]))
            print('Loop {:-4} block {:-2} updated, '
                  'with Error {:.8f}, '
                  'optimum value {:4.6f}, '
                  'Stepsize {:.6f}'.format(
                      t, m, self.error, opti_value2, r))

    def run(self, ERR_BOUND=None, err_iter=None, time_iter=None,
            SILENCE=False, DEBUG=False):
        self.init_flags(ERR_BOUND, err_iter, time_iter)
        # reset paramenters as 0
        self.x.fill(0)
        self.Ax.fill(0)
        self.x_block.fill(0)

        self.block_Cnt = 0
        self.bnd_flag = -1

        start_fun = cuda.Event()
        end_fun = cuda.Event()
        start_event = cuda.Event()
        end_event = cuda.Event()

        time_f = 0
        start_event.record()
        if self.TIME_RCD:
            time_iter[0] = 0

        for t in range(self.ITER_MAX):
            # select mth block
            m = self.index_get(t)
            self.fun_b_k(m)
            self.result_s11 = self.Ax[m] - self.b_k
            # result_s11 = np.sum(Ax, axis=0) - self.b
            # start_fun.record()
            self._mtv(m)
            # end_fun.record()
            # end_fun.synchronize()
            # time_f += start_fun.time_till(end_fun)

            # s14
            self.rx = np.multiply(self.d_ATA[m], self.x_block[m]) -\
                self.result_s13
            self.soft_t = soft_thresholding(self.rx, self.mu)
            # s15
            np.multiply(self.d_ATA_rec[m], self.soft_t, out=self.Bx)
            # check if Bx is near x
            if np.allclose(self.Bx, self.x_block[m], atol=1e-04):
                if not self.IS_BOUNDED:
                    continue
                else:
                    self.bnd_chk(m, ERR_BOUND)
                    if self.bnd_flag == 0:
                        break
                    elif self.bnd_flag == 1:
                        continue
                    else:
                        self.bnd_flag = 4

            self.descent_D = self.Bx - self.x_block[m]
            self._mv(m)
            # stepsize
            r_1 = np.transpose(self.result_s11) @ self.result_s23 +\
                self.mu*(np.linalg.norm(self.Bx, ord=1) -
                         np.linalg.norm(self.x_block[m], ord=1))
            r_2 = np.transpose(self.result_s23) @ self.result_s23
            if r_2 == 0.0:
                print('r_2 is ZERO, could not divide ZERO!')
            else:
                r = np.float64(element_proj(-r_1/r_2, 0, 1))

            self.err_record(err_iter, m, t)
            self.bnd_chk(m, ERR_BOUND)
            if self.bnd_flag == 0:
                break
            elif self.bnd_flag == 1:
                continue

            # x(t+1) = x(t)+r(Bx(t)-x(t))
            self.x_block[m] += r*self.descent_D
            # Ax(t+1)
            self.Ax[m] += r*self.result_s23
            self.debug(DEBUG, t, m, r)

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
        self.x = np.vstack(self.x_block)
        # if not SILENCE:
        #     print(str(time_f/1e3) + ' s.')

        return t_elapsed


class ClassLassoEEC(ClassLasso):
    def run(self, ERR_BOUND=None, err_iter=None, time_iter=None,
            SILENCE=False, DEBUG=False):
        self.init_flags(ERR_BOUND, err_iter, time_iter)
        # reset paramenters as 0
        self.x.fill(0)
        self.Ax.fill(0)
        self.x_block.fill(0)

        self.block_Cnt = 0
        self.bnd_flag = -1

        start_fun = cuda.Event()
        end_fun = cuda.Event()
        start_event = cuda.Event()
        end_event = cuda.Event()

        time_f = 0
        start_event.record()
        if self.TIME_RCD:
            time_iter[0] = 0

        for t in range(self.ITER_MAX):
            # select mth block
            m = self.index_get(t)
            self.fun_b_k(m)
            self.result_s11 = self.Ax[m] - self.b_k
            # result_s11 = np.sum(Ax, axis=0) - self.b
            # start_fun.record()
            self._mtv(m)
            # end_fun.record()
            # end_fun.synchronize()
            # time_f += start_fun.time_till(end_fun)
            self.bnd_chk(m, ERR_BOUND)
            if self.bnd_flag == 0:
                break
            elif self.bnd_flag == 1:
                continue

            # s14
            self.rx = np.multiply(self.d_ATA[m], self.x_block[m]) -\
                self.result_s13
            self.soft_t = soft_thresholding(self.rx, self.mu)
            # s15
            np.multiply(self.d_ATA_rec[m], self.soft_t, out=self.Bx)
            # check if Bx is near x
            if not self.IS_BOUNDED:
                if np.allclose(self.Bx, self.x_block[m], atol=1e-04):
                    continue

            self.descent_D = self.Bx - self.x_block[m]
            self._mv(m)
            # stepsize
            r_1 = np.transpose(self.result_s11) @ self.result_s23 +\
                self.mu*(np.linalg.norm(self.Bx, ord=1) -
                         np.linalg.norm(self.x_block[m], ord=1))
            r_2 = np.transpose(self.result_s23) @ self.result_s23
            if r_2 == 0.0:
                print('r_2 is ZERO, could not divide ZERO!')
            else:
                r = np.float64(element_proj(-r_1/r_2, 0, 1))

            self.err_record(err_iter, m, t)
            # x(t+1) = x(t)+r(Bx(t)-x(t))
            self.x_block[m] += r*self.descent_D
            # Ax(t+1)
            self.Ax[m] += r*self.result_s23
            self.debug(DEBUG, t, m, r)

            self.time_record(time_iter, t, start_event, end_event)

        end_event.record()
        end_event.synchronize()
        if self.TIME_RCD:
            t_elapsed = time_iter[t]
        else:
            t_elapsed = start_event.time_till(end_event) / 1e3

        self.rlt_display(SILENCE, t_elapsed, t)
        self.x = np.vstack(self.x_block)
        # if not SILENCE:
        #     print(str(time_f/1e3) + ' s.')

        return t_elapsed


# CPU & CUDA combined with random block index
class ClassLassoR(ClassLasso):
    def __init__(self, gpu_cal, d_ATA, A, b, mu, BLOCK, ITER_MAX):
        ClassLasso.__init__(self, gpu_cal, d_ATA, A,
                            b, mu, BLOCK, ITER_MAX)
        self.idx_shuffle = np.arange(self.BLOCK)
        self.descript = 'CUDA & CPU random index'

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

        idx_m = self.gpu_cal.MAT_HEIGHT
        idx_n = self.gpu_cal.MAT_WIDTH
        self.s13_gpu = gpuarray.zeros((idx_n, 1), np.float64)
        self.s11_gpu = gpuarray.zeros((idx_m, 1), np.float64)
        self.s23_gpu = gpuarray.zeros((idx_m, 1), np.float64)
        self.d_d_gpu = gpuarray.zeros((idx_n, 1), np.float64)

    # rewrite m @ v and m.T @ v
    def _mtv(self, m):
        self.s11_gpu.set(self.result_s11)
        self._zmvG(self.h, self.s13_gpu, 1, self.gpu_cal.A_b_gpu[m],
                   cublas._CUBLAS_OP['N'], self.s11_gpu)
        self.s13_gpu.get(self.result_s13)

    def _mv(self, m):
        self.d_d_gpu.set(self.descent_D)
        self._zmvG(self.h, self.s23_gpu, 1, self.gpu_cal.A_b_gpu[m],
                   cublas._CUBLAS_OP['T'], self.d_d_gpu)
        self.s23_gpu.get(self.result_s23)

    # z = a * m @ v
    def _zmvG(self, handle, z_gpu, alpha, m_gpu, trans, v_gpu):
        beta = 0
        n, m = m_gpu.shape
        cublas.cublasDgemv(handle, trans, m, n, alpha,
                           m_gpu.gpudata,
                           m, v_gpu.gpudata, 1, beta,
                           z_gpu.gpudata, 1)


class ClassLassoCB_v1EEC(ClassLassoEEC, ClassLassoCB_v1):
    def __init__(self, h, gpu_cal, d_ATA,
                 A, b, mu, BLOCK, ITER_MAX):
        ClassLassoCB_v1.__init__(
            self, h, gpu_cal, d_ATA, A, b, mu, BLOCK, ITER_MAX)


# pure cuBlas
class ClassLassoCB_v2(ClassLasso):
    def __init__(self, h, gpu_cal, d_ATA,
                 A, b, mu, BLOCK, ITER_MAX):
        ClassLasso.__init__(self, gpu_cal, d_ATA, A,
                            b, mu, BLOCK, ITER_MAX)
        self.descript = 'Pure Cublas'
        self.h = h
        idx_m = self.gpu_cal.MAT_HEIGHT
        idx_n = self.gpu_cal.MAT_WIDTH

        self.x_gpu = gpuarray.to_gpu(self.x)
        self.x_block_gpu = [gpuarray.to_gpu(self.x_block[i])
                            for i in range(self.BLOCK)]
        self.Ax_gpu = gpuarray.to_gpu(self.Ax)
        self.b_gpu = gpuarray.to_gpu(self.b)

        self.b_k_gpu = gpuarray.empty_like(self.b_gpu)
        self.rx_gpu = gpuarray.empty_like(self.x_block_gpu[0])
        self.soft_t_gpu = gpuarray.empty_like(self.x_block_gpu[0])
        self.Bx_gpu = gpuarray.empty_like(self.x_block_gpu[0])
        self.s11_gpu = gpuarray.zeros((idx_m, 1), np.float64)
        self.s13_gpu = gpuarray.zeros((idx_n, 1), np.float64)
        self.s23_gpu = gpuarray.zeros((idx_m, 1), np.float64)
        self.d_d_gpu = gpuarray.zeros((idx_n, 1), np.float64)
        self.d_ATA_gpu = self.gpu_cal.d_ATA_gpu
        self.d_ATA_rec_gpu = 1 / self.d_ATA_gpu
        self.err_gpu = gpuarray.empty_like(self.x_block_gpu[0])
        self.close_gpu = gpuarray.zeros(1, np.float64)

        del self.b_k, self.result_s11, self.result_s13,\
            self.result_s21, self.result_s23, self.rx, self.soft_t,\
            self.Bx, self.descent_D

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

        self.allclose = ElementwiseKernel(
            'double *rlt, double *in1, double*in2, double rtol, double atol',
            '''
            if (fabs(in1[i]-in2[i]) - (atol+fabs(in2[i])*rtol) > 0)
            {
                *rlt = 1.0f;
            }
            ''',
            'allclose')

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

    def fun_b_k(self, k):
        cublas.cublasDcopy(self.h, self.b_gpu.size,
                           self.b_gpu.gpudata, 1,
                           self.b_k_gpu.gpudata, 1)
        for i in range(len(self.Ax_gpu)):
            if i != k:
                cublas.cublasDaxpy(self.h, self.Ax_gpu[i].size,
                                   np.float64(-1),
                                   self.Ax_gpu[i].gpudata,
                                   1, self.b_k_gpu.gpudata, 1)

    def r1_get(self, handle, m):
        return cublas.cublasDdot(
            handle, self.s11_gpu.size, self.s11_gpu.gpudata, 1,
            self.s23_gpu.gpudata, 1) +\
                self.mu*(self._l1norm(handle, self.Bx_gpu) -
                         self._l1norm(handle, self.x_block_gpu[m]))

    def err_chk(self, handle, m):
        self._err_array(self.err_gpu, self.s13_gpu,
                        self.x_block_gpu[m], self.mu)
        idx = cublas.cublasIdamax(
            handle, self.err_gpu.size, self.err_gpu.gpudata, 1)
        self.error = np.abs(self.err_gpu[idx].get())[0]

    def bnd_chk(self, handle, m, ERR_BOUND):
        # bnd_flag value
        # 0 break ------ all blocks satisfied criteria
        # 1 continue --- current block satisfies criteria
        # 2 go next ---- current block do not satisfies criteria
        # 4 err_chked -- do not perform bound check again
        if self.bnd_flag == 4:
            self.bnd_flag = 2
        else:
            self.bnd_flag = 2
            if self.IS_BOUNDED:
                if not self.ERR_RCD:
                    self.err_chk(handle, m)
                if self.error < ERR_BOUND:
                    if self.BLOCK - 1 != m:
                        self.block_Cnt += 1
                        self.bnd_flag = 1
                    elif self.block_Cnt != self.BLOCK - 1:
                        self.block_Cnt = 0
                        self.bnd_flag = 1
                    else:
                        self.bnd_flag = 0

    def debug_gpu(self, handle, DEBUG, t, m, r):
        if DEBUG:
            self._zaxpy(self.h, self.s11_gpu, -1, self.b_k_gpu, self.Ax_gpu[m])
            self._zmvG(self.h, self.s13_gpu, 1, self.gpu_cal.A_b_gpu[m],
                       cublas._CUBLAS_OP['N'], self.s11_gpu)
            self.err_chk(handle, m)
            opti_value2 = 0.5*cublas.cublasDdot(
                handle, self.s11_gpu.size, self.s11_gpu.gpudata, 1,
                self.s11_gpu.gpudata, 1) +\
                self.mu*cublas.cublasDasum(
                    handle, self.x_block_gpu[m].size,
                    self.x_block_gpu[m].gpudata, 1)

            print('Loop {:-4} block {:-2} updated, '
                  'with Error {:.8f}, '
                  'optimum value {:4.6f}, '
                  'Stepsize {:.6f}'.format(
                      t, m, self.error, opti_value2, r))

    def err_record_gpu(self, handle, err_iter, err_gpu,
                       s13_gpu, x_block_gpu, mu, m, t):
        if self.ERR_RCD:
            self.err_chk(handle, m)
            err_iter[t] = self.error

    def run(self, ERR_BOUND=None, err_iter=None, time_iter=None,
            SILENCE=False, DEBUG=False):
        self.init_flags(ERR_BOUND, err_iter, time_iter)
        for i in range(self.BLOCK):
            self.x_block_gpu[i].fill(0)

        self.x_gpu.fill(0)
        self.Ax_gpu.fill(0)

        self.block_Cnt = 0
        self.bnd_flag = -1
        self.close_gpu.fill(0)

        start_fun = cuda.Event()
        end_fun = cuda.Event()
        start_event = cuda.Event()
        end_event = cuda.Event()

        time_f = 0
        start_event.record()
        if self.TIME_RCD:
            time_iter[0] = 0

        # cuda.start_profiler()
        for t in range(self.ITER_MAX):
            # select mth block
            m = self.index_get(t)
            self.fun_b_k(m)
            self._zaxpy(self.h, self.s11_gpu, -1, self.b_k_gpu, self.Ax_gpu[m])
            self._zmvG(self.h, self.s13_gpu, 1, self.gpu_cal.A_b_gpu[m],
                       cublas._CUBLAS_OP['N'], self.s11_gpu)
            # s14
            self.d_ATA_gpu[m]._elwise_multiply(
                self.x_block_gpu[m], self.rx_gpu)
            self._axpy(self.h, -1, self.s13_gpu, self.rx_gpu)
            self.zsoft_t(self.soft_t_gpu, self.rx_gpu, self.mu)
            # s15
            self.d_ATA_rec_gpu[m]._elwise_multiply(
                self.soft_t_gpu, self.Bx_gpu)
            # start_fun.record()
            self.allclose(self.close_gpu, self.Bx_gpu, self.x_block_gpu[m],
                          1e-05, 1e-04)
            if self.close_gpu.get()[0] == 0.0:
                if not self.IS_BOUNDED:
                    continue
                else:
                    self.bnd_chk(self.h, m, ERR_BOUND)
                    if self.bnd_flag == 0:
                        break
                    elif self.bnd_flag == 1:
                        continue
                    else:
                        self.bnd_flag = 4
            else:
                self.close_gpu.fill(0)
            # end_fun.record()
            # end_fun.synchronize()
            # time_f += start_fun.time_till(end_fun)

            self._zaxpy(
                self.h, self.d_d_gpu, -1, self.x_block_gpu[m], self.Bx_gpu)
            self._zmvG(
                self.h, self.s23_gpu, 1, self.gpu_cal.A_b_gpu[m],
                cublas._CUBLAS_OP['T'], self.d_d_gpu)

            # stepsize
            r_1g = self.r1_get(self.h, m)
            r_2g = np.square(self._l2norm(self.h, self.s23_gpu))
            if r_2g == 0.0:
                print(t, 'r_2 is ZERO, could not divide ZERO!')
            else:
                r_g = np.float64(element_proj(-r_1g/r_2g, 0, 1))

            self.err_record_gpu(self.h, err_iter, self.err_gpu, self.s13_gpu,
                                self.x_block_gpu[m], self.mu, m, t)
            self.bnd_chk(self.h, m, ERR_BOUND)
            if self.IS_BOUNDED:
                if self.bnd_flag == 0:
                    break
                elif self.bnd_flag == 1:
                    continue

            self._axpy(self.h, r_g, self.d_d_gpu, self.x_block_gpu[m])
            self._axpy(self.h, r_g, self.s23_gpu, self.Ax_gpu[m])
            self.debug_gpu(self.h, DEBUG, t, m, r_g)
            self.time_record(time_iter, t, start_event, end_event)

        end_event.record()
        end_event.synchronize()
        # cuda.stop_profiler()

        if self.TIME_RCD:
            t_elapsed = time_iter[t]
        else:
            t_elapsed = start_event.time_till(end_event) / 1e3

        self.rlt_display(SILENCE, t_elapsed, t)
        self.x = np.vstack(self.x_block)
        # if not SILENCE:
        #     print(str(time_f/1e3) + ' s.')

        return t_elapsed


class ClassLassoCB_v2EEC(ClassLassoCB_v2):
    def run(self, ERR_BOUND=None, err_iter=None, time_iter=None,
            SILENCE=False, DEBUG=False):
        self.init_flags(ERR_BOUND, err_iter, time_iter)
        for i in range(self.BLOCK):
            self.x_block_gpu[i].fill(0)

        self.x_gpu.fill(0)
        self.Ax_gpu.fill(0)

        self.block_Cnt = 0
        self.bnd_flag = -1
        self.close_gpu.fill(0)

        start_fun = cuda.Event()
        end_fun = cuda.Event()
        start_event = cuda.Event()
        end_event = cuda.Event()

        time_f = 0
        start_event.record()
        if self.TIME_RCD:
            time_iter[0] = 0

        # cuda.start_profiler()
        for t in range(self.ITER_MAX):
            # select mth block
            m = self.index_get(t)
            self.fun_b_k(m)
            self._zaxpy(self.h, self.s11_gpu, -1, self.b_k_gpu, self.Ax_gpu[m])
            self._zmvG(self.h, self.s13_gpu, 1, self.gpu_cal.A_b_gpu[m],
                       cublas._CUBLAS_OP['N'], self.s11_gpu)
            # start_fun.record()
            self.bnd_chk(self.h, m, ERR_BOUND)
            if self.IS_BOUNDED:
                if self.bnd_flag == 0:
                    break
                elif self.bnd_flag == 1:
                    continue
            # end_fun.record()
            # end_fun.synchronize()
            # time_f += start_fun.time_till(end_fun)

            # s14
            self.d_ATA_gpu[m]._elwise_multiply(
                self.x_block_gpu[m], self.rx_gpu)
            self._axpy(self.h, -1, self.s13_gpu, self.rx_gpu)
            self.zsoft_t(self.soft_t_gpu, self.rx_gpu, self.mu)
            # s15
            self.d_ATA_rec_gpu[m]._elwise_multiply(
                self.soft_t_gpu, self.Bx_gpu)
            if not self.IS_BOUNDED:
                self.allclose(self.close_gpu, self.Bx_gpu,
                              self.x_block_gpu[m], 1e-05, 1e-04)
                if self.close_gpu.get()[0] == 0.0:
                    continue
                else:
                    self.close_gpu.fill(0)

            self._zaxpy(
                self.h, self.d_d_gpu, -1, self.x_block_gpu[m], self.Bx_gpu)
            self._zmvG(
                self.h, self.s23_gpu, 1, self.gpu_cal.A_b_gpu[m],
                cublas._CUBLAS_OP['T'], self.d_d_gpu)

            # stepsize
            r_1g = self.r1_get(self.h, m)
            r_2g = np.square(self._l2norm(self.h, self.s23_gpu))
            if r_2g == 0.0:
                print(t, 'r_2 is ZERO, could not divide ZERO!')
            else:
                r_g = np.float64(element_proj(-r_1g/r_2g, 0, 1))

            self.err_record_gpu(self.h, err_iter, self.err_gpu, self.s13_gpu,
                                self.x_block_gpu[m], self.mu, m, t)
            self.bnd_chk(self.h, m, ERR_BOUND)
            if self.bnd_flag == 0:
                break
            elif self.bnd_flag == 1:
                continue

            self._axpy(self.h, r_g, self.d_d_gpu, self.x_block_gpu[m])
            self._axpy(self.h, r_g, self.s23_gpu, self.Ax_gpu[m])
            self.debug_gpu(self.h, DEBUG, t, m, r_g)
            self.time_record(time_iter, t, start_event, end_event)

        end_event.record()
        end_event.synchronize()
        # cuda.stop_profiler()

        if self.TIME_RCD:
            t_elapsed = time_iter[t]
        else:
            t_elapsed = start_event.time_till(end_event) / 1e3

        self.rlt_display(SILENCE, t_elapsed, t)
        self.x = np.vstack(self.x_block)
        # if not SILENCE:
        #     print(str(time_f/1e3) + ' s.')

        return t_elapsed
