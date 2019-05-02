
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_WIDTH 32
#define CONSTANT_MASK_SIZE 3000
#define MAX_NUM_THREADS 1024


namespace mxnet
{
namespace op
{

__constant__ float Mask[CONSTANT_MASK_SIZE];

__global__ void
forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

  /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */
  int n, m, c, h, w, p, q;

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  //helps us index the pointers
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define kConstant4d(i3, i2, i1, i0) Mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

  // __shared__ float sharedInput[TILE_WIDTH + K - 1][TILE_WIDTH + K - 1][TILE_WIDTH + K - 1];

  n = blockIdx.x; //idx of images
  m = blockIdx.y; //idx of features
  const int H_Grid = ceil(H_out / (float)TILE_WIDTH);
  const int W_Grid = ceil(W_out / (float)TILE_WIDTH); //round up
  h = blockIdx.z / (H_Grid)*TILE_WIDTH + threadIdx.y; //y idx of output tile
  w = blockIdx.z % (W_Grid)*TILE_WIDTH + threadIdx.x; //x idx of output tile

  if (h >= H_out || w >= W_out)
    return;

  float total = 0.0;

  //num of input feature maps
  // #pragma unroll
  for (c = 0; c < C; c++)
  {
    //height of filter
    // #pragma unroll
    for (p = 0; p < K; p++)
    {
      //width idx of filter
      // #pragma unroll
      for (q = 0; q < K; q++)
      {
        total += x4d(n, c, h + p, w + q) * k4d(m, c, p, q); // kConstant4d(m, c, p, q);
      }
    }
  }

  y4d(n, m, h, w) = total;

  // An example use of these macros:
  // float a = y4d(0,0,0,0)
  // y4d(0,0,0,0) = a

#undef y4d
#undef x4d
#undef k4d
}

__global__ void
unroll_kernel(float *X_unrolled, int size, float *X, int C, int K, int H, int W)
{

#define x4d(i3, i2, i1, i0) X[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define x_u4d(i2, i1, i0) X_unrolled[(i2) * (X_unroll_cols * C * K * K) + (i1) * (X_unroll_cols) + i0]

  //index of thread in unrolled matrix
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= size)
    return;
  //output dimensions
  int H_out = H - K + 1;
  int W_out = W - K + 1;
  int outputSize = H_out * W_out;
  //index into the output images of row and col
  int row = index / outputSize;
  int col = index % outputSize;
  //p and q are used to unroll according to kernel size
  int q = row % K;
  row /= K;
  int p = row % K;
  //c is input featur map index?
  int c = row / K;
  //h and w are of specific X input height and width indices
  int w = col % W_out;
  int h = col / W_out;

  X_unrolled[index] = X[c * (H*W) + (h+p)*W + w+q];
}

void unroll(float* X_unrolled, int size, float* X, int C, int K, int H, int W){
  int gridDim = ceil((1.0*size)/(1.0 * MAX_NUM_THREADS));
  unrollKernel<<<gridDim, MAX_NUM_THREADS>>>(X_unrolled, size, X, C, K, H, W);
}

__global__ void
matrix_multiply_shared(float* A, float* B, float* C, int numAColumns, int numCRows, int numCColumns)
{
  __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];
  
  int by = blockIdx.y;
  int bx = blockIdx.x;
  
  int ty = threadIdx.y;
  int tx = threadIdx.x;
  
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  
  float Pvalue = 0;
  
  for(int m = 0; m < ceil(numAColumns/(float)TILE_WIDTH); m++){

    int idxCol = m*TILE_WIDTH+threadIdx.x;
    if (idxCol < numAColumns)
      subTileM[ty][tx] = A[Row*numAColumns+idxCol];
    else
      subTileM[ty][tx] = 0;

    int idxRow = m*TILE_WIDTH+threadIdx.y;
    if (idxRow < numAColumns)
      subTileN[ty][tx] = B[idxRow*numCColumns + Col];
    else 
      subTileN[ty][tx] = 0;
    
    __syncthreads();
    
    for(int k = 0; k < TILE_WIDTH; k++){
      Pvalue += subTileM[ty][k] * subTileN[k][tx];
    }
    __syncthreads();
    
  }
  if(Row < numCRows && Col < numCColumns)
    C[Row*numCColumns + Col] = Pvalue;
}

void gemm(float* W, float* X_unrolled, float* Y, int CKK, int M, int outputSize)
{
  dim3 gridDim(ceil(1.0*outputSize/TILE_WIDTH), ceil(1.0*M/TILE_WIDTH));
  dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
  matrix_multiply_shared<<<gridDim, blockDim>>>(W, X_unrolled, Y, CKK, M, outputSize);
}

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

  // Use mxnet's CHECK_EQ to do assertions.
  // Remove this assertion when you do your implementation!
  //CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

  // Extract the tensor dimensions into B,M,C,H,W,K

  const int B = x.shape_[0]; //number of output images
  const int M = y.shape_[1]; //number of output feature maps
  const int C = x.shape_[1]; //number of input feature maps
  const int H = x.shape_[2]; //height of output elements
  const int W = x.shape_[3]; //width of output element
  const int K = w.shape_[3]; //dimension of the filters, width and height
  // Set the kernel dimensions
  const int H_out = H - K + 1; // the output after removing the edges
  const int W_out = W - K + 1;

  float* Y = y.dptr_;
  float* X = x.dptr_;
  float* Wptr = w.dptr_;

  // int W_grid = ceil(W_out / (float)TILE_WIDTH); // number of horizontal tiles per output map
  // int H_grid = ceil(H_out / (float)TILE_WIDTH); // number of vertical tiles per output map
  // int Z = H_grid * W_grid;

  // printf("Num Output Feature Maps: %d ", M);
  // printf(" Num Input Feature Maps: %d ", C);
  // printf(" Filter Size: %d \n", K);

  //define sizes
  int outputSize = H_out*W_out;
  int unrollSize = C*K*K*outputSize;
//   int size = C * K * K * outputSize;
  //copy to constant memory
  // cudaMemcpyToSymbol(Mask, w.dptr_, weightSize * sizeof(float));
  // float* X_unrolled = malloc(B * X_unroll_rows * X_unroll_cols * sizeof(float));
  float *X_unrolled;
  cudaMalloc((void **)&X_unrolled, unrollSize * sizeof(float));

  for(int b = B; b--; ){
    int xOffset = b*C*H*W;
    int yOffset = b*M*outputSize;
    unroll(X_unrolled, unrollSize, X + xOffset, C, K, H, W);
    gemm(Wptr, X_unrolled, Y + yOffset, C*K*K, M, outputSize);
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
  CHECK_EQ(0, 1) << "Remove this line and replace it with your implementation.";
}
} // namespace op
} // namespace mxnet

#endif
