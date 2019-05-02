#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define BLOCK_SIZE 1024
#define TILE_WIDTH 32

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__global__ void matrixMultiplyShared(float *A, float *B, float *C, int numAColumns, int numCRows, int numCColumns)
{
  __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;

  float Pvalue = 0;

  for (int m = 0; m < ceil(1.0*numAColumns/TILE_WIDTH); m++) {

    int idxcol = m*TILE_WIDTH+threadIdx.x;
    if (idxcol < numAColumns)
      subTileM[threadIdx.y][threadIdx.x] = A[Row*numAColumns+idxcol];
    else
      subTileM[threadIdx.y][threadIdx.x] = 0;

    int idxrow = m*TILE_WIDTH+threadIdx.y;
    if (idxrow < numAColumns)
      subTileN[threadIdx.y][threadIdx.x] = B[idxrow*numCColumns + Col];
    else
      subTileN[threadIdx.y][threadIdx.x] = 0;


    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; k++)
       Pvalue += subTileM[threadIdx.y][k]*subTileN[k][threadIdx.x];

    __syncthreads();
  }

  if ((Row < numCRows) && (Col < numCColumns)) {
    C[Row*numCColumns+Col] = Pvalue;
  }

}



__global__ void unrollKernel(float* X_unrolled, int size, float* X, int C, int K, int H, int W) {
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    if (index >= size)
        return;
    int H_out = H-K+1;
    int W_out = W-K+1;
    int row = index/(H_out*W_out);
    int col = index%(H_out*W_out);
    int q = row % K;
    row /= K;
    int p = row % K;
    int c = row / K;
    int w = col % W_out;
    int h = col / W_out;
    X_unrolled[index] = X[(c) * (H * W) + (h+p) * (W) + w+q];
}


/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &k)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    // CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = k.shape_[3];
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    float* Y = y.dptr_;
    float* X = x.dptr_;
    float* Kernel = k.dptr_;

    float* X_unrolled;
    int Z = H_out * W_out;
    int size_unrolled = C*K*K*Z;
    cudaMalloc(&X_unrolled, sizeof(float)*size_unrolled);

    dim3 gridDim (ceil(1.0*Z/TILE_WIDTH) , ceil(1.0*M/TILE_WIDTH));
    dim3 blockDim (TILE_WIDTH, TILE_WIDTH);

    for (int b = B; b--; ) {
        unrollKernel<<<ceil(1.0*size_unrolled/BLOCK_SIZE), BLOCK_SIZE>>>(X_unrolled, size_unrolled, X+b*C*H*W, C, K, H, W);

        dim3 gridDim (ceil(1.0*Z/TILE_WIDTH) , ceil(1.0*M/TILE_WIDTH));
        dim3 blockDim (TILE_WIDTH, TILE_WIDTH);
        matrixMultiplyShared<<<gridDim, blockDim>>>(Kernel,  X_unrolled,  Y+b*M*H_out*W_out,  C*K*K,  M,  Z);

    }
    cudaFree(X_unrolled);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
