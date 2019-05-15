import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray

kernel_code_template = """
#define MAT_WIDTH %(MAT_WIDTH)s
#define T_WIDTH_TRANS %(T_WIDTH_TRANS)s
#define T_WIDTH %(T_WIDTH)s
#define MAT_HEIGHT %(MAT_HEIGHT)s
#define T_HEIGHT %(T_HEIGHT)s

__global__ void mul_mat_t_vec(double *result, double *mat, double *vec,
unsigned int index_m){
    const unsigned int row = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int vec_begin = threadIdx.y * T_WIDTH_TRANS;
    const unsigned int mat_begin = index_m * MAT_WIDTH * MAT_HEIGHT;
    double pValue = 0;
    extern __shared__ double sh_pValue[];

    /*//add vec to shared memeory
    extern __shared__ double sh_vec[];
    for (int k = threadIdx.x; k < T_WIDTH_TRANS; k += blockDim.x) {
        sh_vec[threadIdx.y*T_WIDTH_TRANS + k] =\
                vec[vec_begin + k];
    }
    __syncthreads();
    */

    if(row < MAT_WIDTH && vec_begin < MAT_HEIGHT){
        for(int i= 0; (i < T_WIDTH_TRANS) && ((vec_begin+i)<MAT_HEIGHT); i++){
            pValue += mat[mat_begin+(vec_begin+i)*MAT_WIDTH+row]\
                      *vec[vec_begin+i];
            //pValue += mat[mat_begin+(vec_begin+i)*MAT_WIDTH+row]\
            //          *sh_vec[threadIdx.y*T_WIDTH_TRANS + i];
        }
        sh_pValue[threadIdx.x*blockDim.y + threadIdx.y] = pValue;
        //result[row*blockDim.y + threadIdx.y] = pValue;
        __syncthreads();

        for (unsigned int s = blockDim.y/2; s > 0; s >>= 1) {
            if (threadIdx.y < s) {
                sh_pValue[threadIdx.x*blockDim.y + threadIdx.y] += \
                        sh_pValue[threadIdx.x*blockDim.y + threadIdx.y + s];
            }
            __syncthreads();
        }

        if (threadIdx.y == 0) {
            result[row] = sh_pValue[threadIdx.x*blockDim.y];
        }

    }
}


__global__ void mul_mat_t_vec_diffsize(double *result, double *mat,
double *vec, unsigned int index_m){
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int blockxInd;
    __shared__ int blockyInd;
    __shared__ int blockLen;
    __shared__ int mat_begin;
    double pValue = 0;

    if (threadIdx.x == 0) {
        if((blockIdx.x+1)*T_WIDTH_TRANS < MAT_HEIGHT)
            blockLen = T_WIDTH_TRANS;
        else
            blockLen = MAT_WIDTH_ALL % T_WIDTH_TRANS;

        blockxInd = blockIdx.x * T_HEIGHT;
        blockyInd = blockIdx.y * T_WIDTH_TRANS;
        mat_begin = index_m * MAT_WIDTH * MAT_HEIGHT;
    }
    __syncthreads();

    __shared__ sh_vec[T_WIDTH_TRANS];
    if (threadIdx.x < blockLen)
        sh_vec[threadIdx.x] = vec[blockyInd+threadIdx.x];
    __syncthreads();

    int threadxInd = threadIdx.x + blockxInd;
    if (threadxInd < MAT_WIDTH) {
        for (int i = 0; i < T_WIDTH_TRANS; i++)
            pValue += mat[mat_begin+(i+blockyInd)*MAT_WIDTH+threadxInd]\
                    *sh_vec[i];

        atomicAdd(&result[threadxInd], pValue)
    }
}

__global__ void mul_mat_vec(double *result, double *mat, double *vec,
unsigned int index_m){
    const unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    const unsigned int vec_begin = threadIdx.x * T_WIDTH;
    const unsigned int mat_begin = index_m * MAT_WIDTH * MAT_HEIGHT;
    double pValue = 0;

    if((row < MAT_HEIGHT) && (vec_begin < MAT_WIDTH)){
        for(int i= 0; (i < T_WIDTH) && ((vec_begin+i)<MAT_WIDTH); i++){
            pValue += mat[mat_begin+row*MAT_WIDTH+vec_begin+i]\
                      *vec[vec_begin+i];
        }
        result[row*blockDim.x + threadIdx.x] = pValue;
    }
}


__global__ void get_diag_ATA(double *result, double *mat,
unsigned int MAT_WIDTH_ALL){
    const unsigned int col = blockIdx.x * MAT_WIDTH + threadIdx.x;
    const unsigned int mat_begin =\
            blockIdx.x * MAT_WIDTH * MAT_HEIGHT;
    double pValue = 0;
    unsigned int index;

    if((threadIdx.x < MAT_WIDTH) && (col < MAT_WIDTH_ALL)) {
        for(int i = 0; i < MAT_HEIGHT; i++) {
            index = mat_begin+i*MAT_WIDTH+threadIdx.x;
            pValue += mat[index] * mat[index];
        }

        result[blockIdx.x*MAT_WIDTH+threadIdx.x] = pValue;
    }
}
"""

# TODO 
# if matrix A cannot be even blocked,
# append ZEROs to make MAT_WIDTH_ALL % Block = 0

class GPU_Calculation:
    T_WIDTH_TRANS = 64
    T_WIDTH = 128
    T_HEIGHT = 1024

    def __init__(self, A, Block):
        self.Block = Block
        self.MAT_WIDTH_ALL = A.shape[1]
        self.init_cpu_array(A)
        self.init_gpu_array()

        kernel_code = kernel_code_template % {
                'MAT_WIDTH': self.MAT_WIDTH,
                'MAT_HEIGHT': self.MAT_HEIGHT,
                'T_WIDTH_TRANS': self.T_WIDTH_TRANS,
                'T_WIDTH': self.T_WIDTH,
                'T_HEIGHT': self.T_HEIGHT
                }
        mod = SourceModule(kernel_code)
        self.mul_mat_t_vec = mod.get_function("mul_mat_t_vec")
        self.mul_mat_t_vec_diffsize = mod.get_function(
                "mul_mat_t_vec_diffsize")
        self.mul_mat_vec = mod.get_function("mul_mat_vec")
        self.get_diag_ATA = mod.get_function("get_diag_ATA")

        # calculate grid dimension for mul_mat_t_vec_diffsize
        self.blockCols = np.int(
                (MAT_WIDTH+T_HEIGHT-1)/T_HEIGHT)
        self.blockRows = np.int(
                (MAT_HEIGHT+T_WIDTH_TRANS-1)/T_WIDTH_TRANS)

    def init_cpu_array(self, A):
        self.A_b = np.hsplit(A, self.Block)
        self.A_b = np.asarray(self.A_b).astype(np.float64)
        self.MAT_HEIGHT, self.MAT_WIDTH = self.A_b[0].shape
        self.SPLIT_TRANS =\
            (self.MAT_HEIGHT+self.T_WIDTH_TRANS-1) // self.T_WIDTH_TRANS
        self.SPLIT =\
            (self.MAT_WIDTH+self.T_WIDTH-1) // self.T_WIDTH
        self.result_t = np.empty(
                (self.MAT_WIDTH, self.SPLIT_TRANS), np.float64)

        # cpu result when using shared memory and reduce algorithm
        self.result_t_shmem = np.empty(
                (self.MAT_WIDTH, 1), np.float64)

        # cpu result for mul_mat_t_vec_diffsize
        # all set to zero
        self.result_t_diffsize = np.zeros(
                (self.MAT_WIDTH, 1), np.float64)
        self.result = np.empty(
                (self.MAT_HEIGHT, self.SPLIT), np.float64)

    def init_gpu_array(self):
        float64_size = np.dtype(np.float64).itemsize
        self.A_b_gpu = gpuarray.to_gpu(self.A_b)
        self.s11_gpu = cuda.mem_alloc(
                float64_size*self.MAT_HEIGHT)
        self.d_d_gpu = cuda.mem_alloc(
                float64_size*self.MAT_WIDTH)
        self.result_t_gpu = gpuarray.to_gpu(self.result_t)
        self.result_t_shmem_gpu = gpuarray.to_gpu(self.result_t_shmem)

        # gpu array for result_t_diffsize
        self.result_t_diffsize_gpu = gpuarray.to_gpu(
                self.result_t_diffsize)
        self.result_gpu = gpuarray.to_gpu(self.result)

    @property
    def diag_ATA(self):
        d_ATA = np.empty((self.Block, self.MAT_WIDTH, 1), np.float64)
        d_ATA_gpu = gpuarray.to_gpu(d_ATA)
        self.get_diag_ATA(d_ATA_gpu, self.A_b_gpu,
                          np.uint32(self.MAT_WIDTH_ALL),
                          block=(self.MAT_WIDTH, 1, 1),
                          grid=(self.Block, 1, 1))
        d_ATA_gpu.get(d_ATA)
        return d_ATA

    # matrix.T@vector
    # index_m should be unsigned int
    def mat_tmulvec(self, index_m, s11):
        index_m = np.uint32(index_m)
        cuda.memcpy_htod(self.s11_gpu, s11)
        self.mul_mat_t_vec(
                self.result_t_shmem_gpu, self.A_b_gpu, self.s11_gpu, index_m,
                block=(32, self.SPLIT_TRANS, 1), grid=(20, 1, 1),
                shared=8*32*self.SPLIT_TRANS)
                # shared=8*self.SPLIT_TRANS*self.T_WIDTH_TRANS)
        """
        self.result_t_gpu.get(self.result_t)
        result_t = np.sum(self.result_t, axis=1,
                          dtype=np.float64)[:, np.newaxis]
        """
        self.result_t_shmem_gpu.get(self.result_t_shmem)
        return self.result_t_shmem
        """
        return result_t
        """
    # matrix@vector
    def matmulvec(self, index_m, descent_d):
        index_m = np.uint32(index_m)
        cuda.memcpy_htod(self.d_d_gpu, descent_d)
        self.mul_mat_vec(self.result_gpu, self.A_b_gpu, self.d_d_gpu, index_m,
                         block=(self.SPLIT, 200, 1), grid=(1, 16, 1))

        self.result_gpu.get(self.result)
        result = np.sum(
                self.result, axis=1, dtype=np.float64)[:, np.newaxis]

        return result

    # matrix.T@vector for different size matrix
    def mat_tmulvec_diffsize(self, index_m, s11):
        index_m = np.uint32(index_m)
        cuda.memcpy_htod(self.s11_gpu, s11)
        self.mul_mat_t_vec_diffsize(
                self.result_t_diffsize_gpu, self.A_b_gpu, self.s11_gpu,
                index_m,
                block=(self.T_HEIGHT, 1, 1),
                grid=(blockCols, blockRows, 1))
        
        self.result_t_diffsize_gpu.get(self.result_t_diffsize)
        self.result_t_diffsize_gpu.fill(0)
        
        return result_t_diffsize
