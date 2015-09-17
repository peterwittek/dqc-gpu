#include <stdio.h>
#include <cublas.h>

#include "magma.h"

#include "dqc.h"

#define BLOCK_DIM 16 

// Error handling macro
#define CUDA_CHECK(call) \
  if((call) != cudaSuccess) { \
    cudaError_t err = cudaGetLastError(); \
    fprintf (stderr, "CUDA error: %d\n", err ); \
    exit(-1);                                    \
  }

__global__ void columnSquareSum(float *g_idata, float *g_odata, int height, int width)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  float element= (i < width) ? g_idata[i] : 0;
  float sum = element*element;
  float c = 0.0;              
  for (int j = 1; j < height; j++){
    element=(i < width) ? g_idata[j*width+i] : 0;
    float y = element*element - c;  
    float t = sum + y;      
    c = (t - sum) - y;  
    sum = t;            
  }
  g_odata[i]=sum;
}

//M is not transposed after CUBLAS matrix multiplication
__global__ void euclidean(float *odata, float *anorm2, float *bnorm2, float *M, int height, int width)
{
  // read the matrix tile into shared memory
  unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
  unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
  if((xIndex < width) && (yIndex < height))
  {
    unsigned int index=yIndex*width+xIndex;
    odata[index] = anorm2[xIndex]-2*M[index]+bnorm2[yIndex];
  }
}

void calculateNorm2(float *d_Anorm2, float *d_A, int height, int width)
{
  dim3 grid((width+511)/512, 1, 1);
  dim3 threads(512, 1, 1);
  columnSquareSum<<<grid, threads>>>(d_A, d_Anorm2, height, width);
}

float *createGramMatrixGpu(float *A, int nRows, int nCols, float sigma){
  magma_timestr_t magmaStart, magmaTotalStart;
  float* d_A;
  float* d_norm;
  
  magmaTotalStart = get_current_time();
  CUDA_CHECK(cudaMalloc((void**)&d_A, nRows*nCols*sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_norm, nRows*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_A, A, nRows*nCols * sizeof(float), cudaMemcpyHostToDevice));
  
  magmaStart = get_current_time();
  calculateNorm2(d_norm, d_A, nCols, nRows);

  //Calculate the inner products of the data vectors and the weight vectors
  float* d_C;
  CUDA_CHECK( cudaMalloc((void**)&d_C, nRows*nRows*sizeof(float)) );    
  float alpha = 1.0f;float beta = 0.0f;
  cublasSgemm( 'N', 'T', 
              nRows, nRows, nCols, 
              alpha, d_A, nRows, 
                      d_A, nRows, 
               beta, d_C, nRows);
  cublasStatus_t status = cublasGetError();
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("CUBLAS problem: %d\n", status);
  }

  //All components of the vectorized Euclidean distance are available
  float* d_D;
  CUDA_CHECK( cudaMalloc((void**)&d_D, nRows*nRows*sizeof(float)) );
  dim3 grid2((nRows+BLOCK_DIM-1)/BLOCK_DIM, (nRows+BLOCK_DIM-1)/BLOCK_DIM,1);
  dim3 threads2(BLOCK_DIM,BLOCK_DIM,1);
  euclidean<<<grid2, threads2>>>(d_D, d_norm, d_norm, d_C, nRows, nRows);
  cudaFree(d_C);
  cudaFree(d_A);
  cudaFree(d_norm);
  printf("Gram matrix:\t%f\n",  GetTimerValue(magmaStart,get_current_time())/1000.);
  printf("Gram matrix with transfer:\t%f\n",  GetTimerValue(magmaTotalStart,get_current_time())/1000.);
  return d_D;
}
