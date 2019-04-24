
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_WIDTH 16
#define CONSTANT_MASK_SIZE 3000

namespace mxnet
{
namespace op
{

    

__global__ void unroll_Kernel(int C, int H, int W, int K, float* X, float* X_unroll){

    #define x4d(b,m,h,w) x[(b) * (C * H * W) + (m) * (H * W) + (h) * (W) + w]
    #define y4d(m,h,w) unroll_x[(m) * (matrixHeight * matrixWidth) + (h) *(matrixWidth) + w]



     int c, s, h_out, w_out, h_unroll, w_base, p, q;
     int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;//get current thread
     int H_out = H – K + 1; //height of the output tile
     int W_out = W – K + 1; //width of the output tile
     int width = H_out * W_out; //get the total size of the output tile
     if (t < C * width) {
        //get the index of specific output tile we are computing on
        c = threadIndex/ width; 
        s = threadIndex % width;
        //get the index of the specific element within the tile
        h_out = s / H_out; 
        w_out = s % W_out;

        h_unroll = h_out * W_out + w_out;
        w_base = c * K * K;
     for(p = 0; p < K; p++){
     for(q = 0; q < K; q++) {
        w_unroll = w_base + p * K + q;
        X_unroll(h_unroll, w_unroll) = X(c, h_out + p, w_out + q);
     }
     }
    }
}

__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,int numAColumns, int numBRows,int numBColumns, int numCRows,int numCColumns) {

int numRow = blockIdx.x * blockDim.x + threadIdx.x;

int numCol = blockIdx.y * blockDim.y + threadIdx.y;


if ((numRow < numCRows) && (numCol < numCColumns)) {
float value = 0;
for (int i = 0; i < numBRows; i++){
value += A[numRow*numAColumns+i] * B[i*numBColumns+numCol];

}

C[numRow*numCColumns+numCol] = value;
}


}






template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    //CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K

    const int B = x.shape_[0]; //number of input images
    const int M = y.shape_[1]; //number of output feature maps
    const int C = x.shape_[1]; //number of input feature maps/input channels
    const int H = x.shape_[2]; //height of input elements
    const int W = x.shape_[3]; //width of input element
    const int K = w.shape_[3]; //dimension of the filters, width and height

 int H_out = H – K + 1; // height of output elements
 int W_out = W – K + 1; //width of output elements
 int num_threads = C * H_out * W_out; //threads in each block, each output tile * number of channels
 int num_blocks = ceil((C * H_out * W_out) / CUDA MAX_NUM_THREADS); //get number of blocks

 float* X_unroll;
 cudaMalloc((void **) &X_unroll, (B*C * K * K* H_out * W_out)*sizeof(float));

 dim3 blockDim(num_threads, 1, 1);
 dim3 gridDim(B, ceil((C * H_out * W_out) / TILE_WIDTH), 1); 
//call kernal
 unroll_Kernel<<<gridDrim, blockDim>>>(C, H, W, K, x.dptr_, X_unroll);

 dim3 GridDim(ceil(numCRows/30.0), ceil(numCColumns /30.0), 1);
 dim3 BlockDim(30, 30, 1); 

 matrixMultiply<<<GridDim, BlockDim>>>(deviceA, deviceB, y.dptr_, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

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