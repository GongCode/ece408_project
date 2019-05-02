#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define BLOCK_SIZE 1024
#define TILE_WIDTH 16

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__global__ void matrixMultiplyShared(float *__restrict__ Kernel, float *__restrict__ X, float *__restrict__ Y, int M, int C, int H, int W, int K)
{
  // numARows = numCRows
  // numBRows = numAColumns
  // numBColumns = numCColumns

  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  int numAColumns = C * K * K;
  int numCRows = M;
  int H_out = H - K + 1;
  int W_out = W - K + 1;
  int numCColumns = H_out * W_out;

  __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;

  float Pvalue = 0;
  int index, row, col, q, p, c, w, h;

  float *temp;

#pragma unroll
  for (int tileIx = 0; tileIx < ceil(1.0 * numAColumns / TILE_WIDTH); tileIx++)
  {

    int idxcol = tileIx * TILE_WIDTH + threadIdx.x;
    if (idxcol < numAColumns)
      subTileM[threadIdx.y][threadIdx.x] = Kernel[Row * numAColumns + idxcol];
    else
      subTileM[threadIdx.y][threadIdx.x] = 0;

    int idxrow = tileIx * TILE_WIDTH + threadIdx.y;
    if (idxrow < numAColumns)
    {
      index = idxrow * numCColumns + Col;
      row = index / (H_out * W_out);
      col = index % (H_out * W_out);
      q = row % K;
      row /= K;
      p = row % K;
      c = row / K;
      w = col % W_out;
      h = col / W_out;
      subTileN[threadIdx.y][threadIdx.x] = X[(c) * (H * W) + (h + p) * (W) + w + q];
    }
    else
      subTileN[threadIdx.y][threadIdx.x] = 0;

    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE_WIDTH; k++)
      Pvalue += subTileM[threadIdx.y][k] * subTileN[k][threadIdx.x];

    __syncthreads();
  }

  if ((Row < numCRows) && (Col < numCColumns))
  {
    Y[Row * numCColumns + Col] = Pvalue;
  }
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
  float *Y = y.dptr_;
  float *X = x.dptr_;
  float *Kernel = k.dptr_;

  dim3 gridDim(ceil(1.0 * H * W / TILE_WIDTH), ceil(1.0 * M / TILE_WIDTH));
  dim3 blockDim(TILE_WIDTH, TILE_WIDTH);

  for (int b = B; b--;)
  {
    matrixMultiplyShared<<<gridDim, blockDim>>>(Kernel, X + b * C * H * W, Y + b * M * H_out * W_out, M, C, H, W, K);
  }
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
  CHECK_EQ(0, 1) << "Remove this line and replace it with your implementation.";
}
} // namespace op
} // namespace mxnet

#endif
