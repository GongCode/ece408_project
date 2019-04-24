
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_WIDTH 16
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
unroll_kernel(int C, int H, int W, int H_out, int W_out, int X_unroll_cols, int K, float* X, float* X_unrolled)
{

#define x4d(i3, i2, i1, i0) X[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define x_u4d(i2, i1, i0) X_unrolled[(i2) * (X_unroll_cols*C*K*K) + (i1) * (X_unroll_cols) + i0]

  int c, s, h_out, w_out, h_unroll, w_unroll, w_base, p, q;
  int t = blockIdx.x * MAX_NUM_THREADS + threadIdx.x;
  int n = blockIdx.y; // index of images
  if (t<C*X_unroll_cols){
    c = t/X_unroll_cols;
    s = t%X_unroll_cols;
    h_out = s/W_out;
    w_out = s%W_out;
    w_unroll = h_out * W_out + w_out;
    w_base = c * K * K;
    for(p = 0; p < K; p++){
    for(q = 0; q < K; q++) {
      h_unroll = w_base + p * K + q;
      x_u4d(n, h_unroll, w_unroll) = x4d(n, c, h_out + p, w_out + q);
      }
    }
  }

}

__global__ void
matrix_multiply(float* Y, float* X_unrolled, float* K, int K_unroll_rows, int K_unroll_cols, int X_unroll_rows, int X_unroll_cols, int Y_unroll_rows, int Y_unroll_cols)
{
  float value = 0;
  int row = blockIdx.x*blockDim.x + threadIdx.x;
  int column = blockIdx.y*blockDim.y + threadIdx.x;

  if(row < K_unroll_rows && column < X_unroll_cols){
    for (int i = 0; i < K_unroll_cols; i++){
      value += K[row * K_unroll_cols + i] * X_unrolled[(X_unroll_cols*X_unroll_rows)*blockIdx.z + i*X_unroll_cols + column];
    }
    Y[(Y_unroll_rows*Y_unroll_cols)*blockIdx.z + row*Y_unroll_cols + column] = value;
  }
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
    printf("Reached");
    // Set the kernel dimensions
    const int H_out = H - K + 1; // the output after removing the edges
    const int W_out = W - K + 1;

    int W_grid = ceil(W_out / (float)TILE_WIDTH); // number of horizontal tiles per output map
    int H_grid = ceil(H_out / (float)TILE_WIDTH); // number of vertical tiles per output map
    int Z = H_grid * W_grid;

    printf("Num Output Feature Maps: %d ", M);
    printf(" Num Input Feature Maps: %d ", C);
    printf(" Filter Size: %d ", K);

    //Dimensions of matrices to support unrolling
    int X_unroll_rows = C*K*K;
    int X_unroll_cols = H_out * W_out;
    int K_unroll_rows = M;
    int K_unroll_cols = C*K*K;
    int Y_unroll_rows = M;
    int Y_unroll_cols = H_out * W_out;

    //define weight size
    int weightSize = M * C * K * K;
    //copy to constant memory
    cudaMemcpyToSymbol(Mask, w.dptr_, weightSize * sizeof(float));
    // float* X_unrolled = malloc(B * X_unroll_rows * X_unroll_cols * sizeof(float));
    float* X_unrolled;
    cudaMalloc((void**)&X_unrolled, B * X_unroll_rows * X_unroll_cols * sizeof(float));
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(B, M, Z); //num of output images, number of output feature maps, total tiles

    dim3 grid_unroll(ceil(C*H_out*W_out / MAX_NUM_THREADS), B, 1);
    dim3 block_unroll(MAX_NUM_THREADS,1,1);
    unroll_kernel<<<grid_unroll, block_unroll>>>(C, H, W, H_out, W_out, X_unroll_cols, K, x.dptr_, X_unrolled);
    cudaDeviceSynchronize();


    dim3 block_mm(16,16,1);
    dim3 grid_mm(ceil(Y_unroll_rows/block_mm.x), ceil(Y_unroll_cols/block_mm.y), B);
    matrix_multiply<<<grid_mm, block_mm>>>(y.dptr_, X_unrolled, w.dptr_, K_unroll_rows, K_unroll_cols, X_unroll_rows, X_unroll_cols, Y_unroll_rows, Y_unroll_cols);


    // Call the kernel
    // forward_kernel<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);

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
