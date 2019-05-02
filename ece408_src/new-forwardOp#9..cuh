
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>


#define TILE_WIDTH_1 8
#define TILE_WIDTH 16
#define CONSTANT_MASK_SIZE 3000

namespace mxnet
{
namespace op
{

__global__ void
forward_kernel_1(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
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

    n = blockIdx.x; //idx of images
    m = blockIdx.y; //idx of features
    const int H_Grid = ceil(H_out / (float)TILE_WIDTH_1);
    const int W_Grid = ceil(W_out / (float)TILE_WIDTH_1); //round up
    h = blockIdx.z / (H_Grid)*TILE_WIDTH_1 + threadIdx.y; //y idx of output tile
    w = blockIdx.z % (W_Grid)*TILE_WIDTH_1 + threadIdx.x; //x idx of output tile

    if (h >= H_out || w >= W_out)
        return;

    float total = 0.0;

    //num of input feature maps
    for (c = 0; c < C; c++)
    {
        //height of filter
        for (p = 0; p < K; p++)
        {
            //width idx of filter
            for (q = 0; q < K; q++)
            {
                total += x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
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
    for (c = 0; c < C; c++)
    {
        //height of filter
        for (p = 0; p < K; p++)
        {
            //width idx of filter
            for (q = 0; q < K; q++)
            {
                total += x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
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

    int W_grid = ceil(W_out / (float)TILE_WIDTH); // number of horizontal tiles per output map
    int H_grid = ceil(H_out / (float)TILE_WIDTH); // number of vertical tiles per output map
    int Z = H_grid * W_grid;

    printf("Num Output Feature Maps: %d ", M);
    printf(" Num Input Feature Maps: %d ", C);
    printf(" Filter Size: %d ", K);

    if(C == 1){
        dim3 blockDim(TILE_WIDTH_1, TILE_WIDTH_1, 1);
        dim3 gridDim(B, M, Z); //num of output images, number of output feature maps, total tiles
        // Call the kernel
        forward_kernel_1<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);
    }else{
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        dim3 gridDim(B, M, Z); //num of output images, number of output feature maps, total tiles
        // Call the kernel
        forward_kernel<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);

    }

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(B, M, Z); //num of output images, number of output feature maps, total tiles

    // Call the kernel
    forward_kernel<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);

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