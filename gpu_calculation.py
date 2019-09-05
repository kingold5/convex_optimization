import numpy as np
from jinja2 import Template
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray


kernel_code_template = Template("""
#define MAT_WIDTH {{MAT_WIDTH}}
#define MAT_HEIGHT {{MAT_HEIGHT}}
#define T_WIDTH_TRANS {{T_WIDTH_TRANS}}
#define T_WIDTH {{T_WIDTH}}
#define T_HEIGHT {{T_HEIGHT}}
#define TYPE {{TYPE}}
#define BLOCK_DIM 16
//#include <math.h>
#include <pycuda-complex.hpp>


__global__ void mul_mat_t_vec_diffsize(TYPE *result, TYPE *mat,
TYPE *vec, const int mat_begin, const int height, const int width){
    __shared__ int blockxInd;
    __shared__ int blockyInd;
    __shared__ int blockLen;
    TYPE pValue = 0.0;

    if (threadIdx.x == 0) {
        if((blockIdx.y+1)*T_WIDTH_TRANS <= height){
            blockLen = T_WIDTH_TRANS;
        }
        else blockLen = height % T_WIDTH_TRANS;

        blockxInd = blockIdx.x * T_HEIGHT;
        blockyInd = blockIdx.y * T_WIDTH_TRANS;
    }
    __syncthreads();

    __shared__ TYPE sh_vec[T_WIDTH_TRANS];
    int vecInd = threadIdx.x;
    while (vecInd < blockLen) {
        sh_vec[vecInd] = vec[blockyInd+vecInd];
        vecInd += T_HEIGHT;
    }
    __syncthreads();

    int threadxInd = threadIdx.x + blockxInd;
    if (threadxInd < width) {
        for (int i = 0; i < blockLen; i++)
            pValue += mat[mat_begin+(i+blockyInd)*width+threadxInd]\
                    *sh_vec[i];

        //atomicAdd(result + threadxInd, pValue);
        result[threadxInd*gridDim.y+blockIdx.y] = pValue;
    }
}


__global__ void mul_mat_vec_diffsize(TYPE *result, TYPE *mat,
TYPE *vec, const int mat_begin, const int height, const int width){
    __shared__ int blockxInd;
    __shared__ int blockyInd;
    __shared__ int blockLen;
    TYPE pValue = 0;

    if (threadIdx.y == 0) {
        if((blockIdx.x+1)*T_WIDTH <= width)
            blockLen = T_WIDTH;
        else blockLen = width % T_WIDTH;

        blockxInd = blockIdx.x * T_WIDTH;
        blockyInd = blockIdx.y * T_HEIGHT;
    }
    __syncthreads();

    __shared__ TYPE sh_vec[T_WIDTH];
    int vecInd = threadIdx.y;
    while (vecInd < blockLen) {
        sh_vec[vecInd] = vec[blockxInd+vecInd];
        vecInd += T_HEIGHT;
    }
    __syncthreads();

    int threadyInd = threadIdx.y + blockyInd;
    if (threadyInd < height) {
        for (int i = 0; i < blockLen; i++)
            pValue += mat[mat_begin+threadyInd*width+blockxInd+i]\
                    *sh_vec[i];

        result[threadyInd*gridDim.x+blockIdx.x] = pValue;
    }
}


__global__ void mat_transpose(TYPE *result, TYPE *mat,
const int mat_begin, const int height, const int width) {
    __shared__ TYPE block[BLOCK_DIM][BLOCK_DIM+1];

    int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
    if ((yIndex < height) && (xIndex < width)) {
        int index_in = mat_begin + yIndex * width + xIndex;
        block[threadIdx.y][threadIdx.x] = mat[index_in];
    }

    __syncthreads();

    xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
    yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
    if ((yIndex < width) && (xIndex < height)) {
        int index_out = yIndex * height + xIndex;
        result[index_out] = block[threadIdx.x][threadIdx.y];
    }
}


__global__ void get_diag_ATA(TYPE *result, TYPE *mat,
unsigned int index_m) {
    __shared__ int blockxInd;

    if (threadIdx.x == 0) {
        blockxInd = blockIdx.x * blockDim.x;
    }
    __syncthreads();

    TYPE pValue = 0;
    unsigned int k;

    int threadxInd = blockxInd + threadIdx.x;
    if (threadxInd < MAT_WIDTH) {
        for(int i = 0; i < MAT_HEIGHT; i++) {
            k = i*MAT_WIDTH+threadxInd;
            pValue += pow(mat[k], 2.0f);
        }

        result[threadxInd] = pValue;
    }
}


__global__ void get_diag_ATA_c(pycuda::complex<double> *result, pycuda::complex<double> *mat,
unsigned int index_m){
    __shared__ int blockxInd;

    if (threadIdx.x == 0) {
        blockxInd = blockIdx.x * blockDim.x;
    }
    __syncthreads();

    pycuda::complex<double> pValue(0, 0);
    int k;

    int threadxInd = blockxInd + threadIdx.x;
    if (threadxInd < MAT_WIDTH) {
        for (int i = 0; i < MAT_HEIGHT; i++) {
            k = i + threadxInd * MAT_HEIGHT;
            pValue += conj(mat[k]) * mat[k];
        }

        result[threadxInd] = pValue;
    }
}
""")


class GPU_Calculation:

    T_WIDTH_TRANS = 64
    T_WIDTH = 64
    T_HEIGHT = 512
    TYPE = 'double'

    def __init__(self, A, Block):
        if A.dtype == 'float64':
            self.TYPE = 'double'
        elif A.dtype == 'float32':
            self.TYPE = 'float'

        self.Block = Block
        self.MAT_WIDTH_ALL = A.shape[1]
        self.init_cpu_array(A)
        self.init_gpu_array()

        kernel_code = kernel_code_template.render(
            MAT_WIDTH=self.MAT_WIDTH,
            MAT_HEIGHT=self.MAT_HEIGHT,
            T_WIDTH_TRANS=self.T_WIDTH_TRANS,
            T_WIDTH=self.T_WIDTH,
            T_HEIGHT=self.T_HEIGHT,
            TYPE=self.TYPE
            )

        mod = SourceModule(kernel_code)
        self.mul_mat_t_vec_diffsize = mod.get_function(
            "mul_mat_t_vec_diffsize")
        self.mul_mat_vec_diffsize = mod.get_function(
            "mul_mat_vec_diffsize")
        self.get_diag_ATA = mod.get_function("get_diag_ATA")
        self.get_diag_ATA_c = mod.get_function("get_diag_ATA_c")
        # self.mat_transpose = mod.get_function("mat_transpose")

    def init_cpu_array(self, A):
        self.A_b = A.transpose().reshape(self.Block, -1, A.shape[0]).swapaxes(-1, -2)
        self.MAT_HEIGHT, self.MAT_WIDTH = self.A_b[0].shape

        # -----------------------------------------------------------
        # set grid for cuda gpu

        # grid dimension for matrix.T@vector diffsize cuda
        self.block_cols_t = np.int(
            (self.MAT_WIDTH+self.T_HEIGHT-1)/self.T_HEIGHT)
        self.block_rows_t = np.int(
            (self.MAT_HEIGHT+self.T_WIDTH_TRANS-1)/self.T_WIDTH_TRANS)

        # grid dimension for matrix@vector diffsize cuda
        self.block_cols = np.int(
            (self.MAT_WIDTH+self.T_WIDTH-1)/self.T_WIDTH)
        self.block_rows = np.int(
            (self.MAT_HEIGHT+self.T_HEIGHT-1)/self.T_HEIGHT)

        # grid dimension for matrix.T diffsize
        self.block_cols_mt = np.int((self.MAT_WIDTH + 16 - 1)/16)
        self.block_rows_mt = np.int((self.MAT_HEIGHT + 16 - 1)/16)

        # grid dimension for matrix@vector diffsize after transpose
        # self.block_cols_at = np.int(
        #     (self.MAT_HEIGHT+self.T_HEIGHT-1)/self.T_HEIGHT)
        # self.block_rows_at = np.int(
        #     (self.MAT_WIDTH+self.T_WIDTH_TRANS-1)/self.T_WIDTH_TRANS)

        # -------------------------------------------------------------
        # init gpuarray to store temporal result

        # cpu result for mul_mat_t_vec_diffsize
        # all set to zero cuz using atomicAdd()
        # self.result_t_diffsize = np.zeros(
        #     (self.MAT_WIDTH, 1), np.float64)

        # cpu result for matrix.T@vector diffsize without atomicAdd()
        self.result_t_diffsize = np.zeros(
            (self.MAT_WIDTH, self.block_rows_t), np.float64)

        # cpu result for matrix@vector diffsize without atomicAdd()
        self.result_diffsize = np.zeros(
            (self.MAT_HEIGHT, self.block_cols), np.float64)

        # cpu result for matrix@vector diffsize after transpose
        # self.result_at_diffsize = np.zeros(
        #     (self.MAT_HEIGHT, self.block_rows_at), np.float64)

    def init_gpu_array(self):
        float64_size = np.dtype(np.float64).itemsize
        if self.A_b.dtype == 'float64':
            self.A_b_gpu = gpuarray.to_gpu(self.A_b.copy())
        elif self.A_b.dtype == 'complex128':
            self.A_b_gpu = gpuarray.to_gpu(self.A_b.swapaxes(-1, -2).copy())
        # self.A_b_cw_gpu = gpuarray.to_gpu(self.A_b_cw)
        self.s11_gpu = cuda.mem_alloc(float64_size*self.MAT_HEIGHT)
        self.d_d_gpu = cuda.mem_alloc(float64_size*self.MAT_WIDTH)

        # result for matrix.T@vector diffsize
        self.result_t_diffsize_gpu = gpuarray.to_gpu(
            self.result_t_diffsize)

        # result for matrix@vector diffsize
        self.result_diffsize_gpu = gpuarray.to_gpu(
            self.result_diffsize)

        # result for A_b[index_m].Transpose
        # self.A_b_T_gpu = gpuarray.empty(
        #         (self.MAT_WIDTH, self.MAT_HEIGHT), np.float64)

        # result for matrix@vector diffsize after transpose
        # self.result_at_diffsize_gpu = gpuarray.to_gpu(
        #     self.result_at_diffsize)

    def diag_ATA(self):
        self.d_ATA = np.empty((self.Block, self.MAT_WIDTH, 1), np.float64)
        self.d_ATA_gpu = gpuarray.to_gpu(self.d_ATA)
        block_threads = 1024
        block_cols_d = np.int((self.MAT_WIDTH+block_threads-1) /
                              block_threads)

        for index in range(self.Block):
            self.get_diag_ATA(
                self.d_ATA_gpu[index], self.A_b_gpu[index], np.int32(index),
                block=(block_threads, 1, 1),
                grid=(block_cols_d, 1, 1))

        self.d_ATA_gpu.get(self.d_ATA)

    def diag_ATA_c(self):
        self.d_ATA_c = np.empty((self.Block, self.MAT_WIDTH, 1), np.complex128)
        self.d_ATA_c_gpu = gpuarray.to_gpu(self.d_ATA_c)
        block_threads = 1024
        block_cols_d = np.int((self.MAT_WIDTH+block_threads-1) /
                              block_threads)

        for index in range(self.Block):
            self.get_diag_ATA_c(
                self.d_ATA_c_gpu[index], self.A_b_gpu[index], np.int32(index),
                block=(block_threads, 1, 1),
                grid=(block_cols_d, 1, 1))

        self.d_ATA_c_gpu.get(self.d_ATA_c)

    # matrix.T@vector for different size matrix for cuda
    def mat_tMulVec_DiffSize(self, s13, index_m, s11):
        cuda.memcpy_htod(self.s11_gpu, s11)
        mat_begin = np.int32(index_m * self.MAT_HEIGHT * self.MAT_WIDTH)

        self.mul_mat_t_vec_diffsize(
                self.result_t_diffsize_gpu, self.A_b_gpu, self.s11_gpu,
                mat_begin, np.int32(self.MAT_HEIGHT), np.int32(self.MAT_WIDTH),
                block=(self.T_HEIGHT, 1, 1),
                grid=(self.block_cols_t, self.block_rows_t, 1))

        # call cpu to calculate instead of atomicadd()
        self.result_t_diffsize_gpu.get(self.result_t_diffsize)
        np.sum(self.result_t_diffsize, axis=1,
               dtype=np.float64, out=s13, keepdims=True)

    # matrix@vector for different size matrix for cuda
    def matMulVec_DiffSize(self, s23, index_m, descent_d):
        cuda.memcpy_htod(self.d_d_gpu, descent_d)
        mat_begin = np.int32(index_m * self.MAT_HEIGHT * self.MAT_WIDTH)

        self.mul_mat_vec_diffsize(
            self.result_diffsize_gpu, self.A_b_gpu, self.d_d_gpu,
            mat_begin, np.int32(self.MAT_HEIGHT), np.int32(self.MAT_WIDTH),
            block=(1, self.T_HEIGHT, 1),
            grid=(self.block_cols, self.block_rows, 1))

        self.result_diffsize_gpu.get(self.result_diffsize)
        np.sum(self.result_diffsize, axis=1, out=s23,
               dtype=np.float64, keepdims=True)[:, np.newaxis]

"""
    # matrix@vector for different size matrix but transpose first
    def matMulVec_DST(self, index_m, descent_d):
        cuda.memcpy_htod(self.d_d_gpu, descent_d)
        self.matTranspose(index_m)

        self.mul_mat_t_vec_diffsize(
            self.result_at_diffsize_gpu, self.A_b_T_gpu, self.d_d_gpu,
            np.int32(0), np.int32(self.MAT_WIDTH), np.int32(self.MAT_HEIGHT),
            block=(self.T_HEIGHT, 1, 1),
            grid=(self.block_cols_at, self.block_rows_at, 1))

        self.result_at_diffsize_gpu.get(self.result_at_diffsize)
        return np.sum(self.result_at_diffsize, axis=1,
                      dtype=np.float64)[:, np.newaxis]

    # matrix.T for diffsize
    def matTranspose(self, index_m):
        mat_begin = np.int32(index_m * self.MAT_HEIGHT * self.MAT_WIDTH)

        self.mat_transpose(
                self.A_b_T_gpu, self.A_b_gpu, mat_begin,
                np.int32(self.MAT_HEIGHT),
                np.int32(self.MAT_WIDTH),
                block=(16, 16, 1),
                grid=(self.block_cols_mt, self.block_rows_mt, 1))
"""
