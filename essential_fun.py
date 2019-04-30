# -*- coding: utf-8 -*-

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray

kernel_code_template1 = """
#define MAT_WIDTH %(MAT_WIDTH)s
#define THREAD_WIDTH %(THREAD_WIDTH)s
#define MAT_HEIGHT %(MAT_HEIGHT)s
__global__ void mul_mat_t_vec(double *result, double *mat, double *vec){
    const unsigned int row = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int vec_begin = threadIdx.y * THREAD_WIDTH;
    double pValue = 0;
    /*//add vec to shared memeory
    __shared__ double sh_vec[blockDim.x][THREAD_WIDTH];
    for(int i=0; i < THREAD_WIDTH; i++)
    {
        sh_vec[threadIdx.x][i] = vec[vec_begin+i];
    }
    __syncthreads();
    */
    if(row < MAT_WIDTH && vec_begin < MAT_HEIGHT){
        for(int i= 0; (i < THREAD_WIDTH) && ((vec_begin+i)<MAT_HEIGHT); i++){
            pValue += mat[(vec_begin+i)*MAT_WIDTH+row]*vec[vec_begin+i];
        }
        result[row*blockDim.y + threadIdx.y] = pValue;
    }
}
"""
kernel_code_template2 = """
#define MAT_WIDTH %(MAT_WIDTH)s
#define THREAD_WIDTH %(THREAD_WIDTH)s
#define MAT_HEIGHT %(MAT_HEIGHT)s
__global__ void mul_mat_vec(double *result, double *mat, double *vec){
    const unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    const unsigned int vec_begin = threadIdx.x * THREAD_WIDTH;
    double pValue = 0;

    if((row < MAT_HEIGHT) && (vec_begin < MAT_WIDTH)){
        for(int i= 0; (i < THREAD_WIDTH) && ((vec_begin+i)<MAT_WIDTH); i++){
            pValue += mat[row*MAT_WIDTH+vec_begin+i]*vec[vec_begin+i];
        }
        result[row*blockDim.x + threadIdx.x] = pValue;
    }
}
"""

kernel_code_template3 = """
#define MAT_WIDTH %(MAT_WIDTH)s
#define THREAD_WIDTH %(THREAD_WIDTH)s
#define MAT_HEIGHT %(MAT_HEIGHT)s
__global__ void mul_mat_vec1(double *result, double *mat, double *vec){
    const unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    const unsigned int vec_begin = threadIdx.x * THREAD_WIDTH;
    double pValue = 0;

    if((row < MAT_HEIGHT) && (vec_begin < MAT_WIDTH)){
        for(int i= 0; (i < THREAD_WIDTH) && ((vec_begin+i)<MAT_WIDTH); i++){
            pValue += mat[row*MAT_WIDTH+vec_begin+i]*vec[vec_begin+i];
        }
        result[row*blockDim.x + threadIdx.x] = pValue;
    }
}
"""

def soft_thresholding(tensor, threshold):
    return np.sign(tensor)*np.maximum(np.abs(tensor)-threshold, 0)


#project vec onto [lower_bound, upper_bound]
def element_proj(vec, lower_bound, upper_bound):
    return np.maximum(np.minimum(vec, upper_bound), lower_bound)


#calculate error
def error_crit(grad_fx, x, mu):
    projection = element_proj(grad_fx-x, -mu, mu)
    #l2 norm
    #return np.linalg.norm(grad_fx-projection)
    #l infinity norm
    return np.max(np.abs(grad_fx-projection))


def fun_s12(A_b, s11):
    T_WIDTH = 64
    A_b_gpu = gpuarray.to_gpu(A_b.astype(np.float64))
    s11_gpu = gpuarray.to_gpu(s11.astype(np.float64))
    SPLIT = (A_b.shape[0]+T_WIDTH-1)//T_WIDTH

    #SPLIT = (A_b.shape[1]+T_WIDTH-1)//T_WIDTH
    result = np.empty((A_b.shape[1], SPLIT), np.float64)

    #result = np.empty((A_b.shape[0], SPLIT), np.float64)
    result_gpu = gpuarray.to_gpu(result)
    kernel_code = kernel_code_template1 % {
            'MAT_WIDTH': A_b.shape[1],
            'MAT_HEIGHT': A_b.shape[0],
            'THREAD_WIDTH': T_WIDTH
            }
    mod = SourceModule(kernel_code)
    fun_s12_gpu = mod.get_function("mul_mat_t_vec")
    fun_s12_gpu(result_gpu, A_b_gpu, s11_gpu, block=(32, SPLIT, 1),
            grid=(20, 1, 1))
    #fun_s12_gpu(result_gpu, A_b_gpu, s11_gpu, block=(SPLIT, 32, 1),
    #        grid=(1, 20, 1))
    result = result_gpu.get()
    result = np.sum(result, axis=1, dtype=np.float64)[:, np.newaxis]
    #print(result - A_b@s11)
    return result

#calculate blockwise diagonal of matrix A.transpose()*A
def fun_diag_ATA(A_b):
    diag_ATA = []
    for i in range(len(A_b)):
        diag_ATA.append(np.sum(np.power(A_b[i], 2), axis=0)[:, np.newaxis])
    diag_ATA = np.asarray(diag_ATA)
    return diag_ATA

def fun_s22(A_b, descent_d):
    T_WIDTH = 128
    A_b_gpu = gpuarray.to_gpu(A_b.astype(np.float64))
    descent_d_gpu = gpuarray.to_gpu(descent_d.astype(np.float64))
    SPLIT = (A_b.shape[1]+T_WIDTH-1)//T_WIDTH
    result = np.empty((A_b.shape[0], SPLIT), np.float64)
    result_gpu = gpuarray.to_gpu(result)
    kernel_code = kernel_code_template2 % {
            'MAT_WIDTH': A_b.shape[1],
            'MAT_HEIGHT': A_b.shape[0],
            'THREAD_WIDTH': T_WIDTH
            }

    mod = SourceModule(kernel_code)
    fun_s22_gpu = mod.get_function("mul_mat_vec")
    fun_s22_gpu(result_gpu, A_b_gpu, descent_d_gpu,
            block=(SPLIT, 200, 1), grid=(1, 16, 1))
    result = result_gpu.get()
    result = np.sum(result, axis=1, dtype=np.float64)[:, np.newaxis]
    return result


# needs to update
# calculate b_k, excluded K_th block
def fun_b_k(Ax, b, k):
    if len(Ax) <= 1:
        return b
    else:
        result = np.zeros((len(Ax[0]), 1))
        for i in range(len(Ax)):
            if i != k:
                result += Ax[i]
        return -result + b


"""
def fun_dd_p(A_bp, descent_d):
    dd_p = []
    count = 0
    for i in range(len(A_bp)):
        dd_p.append(descent_d[count: count+len(A_bp[i][0]), :]) 
        count += len(A_bp[i][0])
    return dd_p
"""
