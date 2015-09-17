#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cublas.h>

#include "magma.h"

#include "dqc.h"

// Error handling macro
#define CUDA_CHECK(call) \
  if((call) != cudaSuccess) { \
    cudaError_t err = cudaGetLastError(); \
    fprintf (stderr, "CUDA error: %d\n", err ); \
    exit(-1);                                    \
  }

CUcontext context;                                                        \

void checkMatrix(float *d_data, int height, int width){
  float *data=(float *)malloc(height*width*sizeof(float));
  CUDA_CHECK( cudaMemcpy(data, d_data, height*width*sizeof(float), cudaMemcpyDeviceToHost) );
  for (int i=0; i<height; ++i){
    for (int j=0; j<width; ++j){
      printf("%f ", data[i*width+j]);
    }
    printf("\n");
  }
  free(data);
}

void initializeGpu()
{
  CUdevice  dev;                                                        \
  if( CUDA_SUCCESS != cuInit( 0 ) ) {                                        \
    fprintf(stderr, "CUDA: Not initialized\n" ); exit(-1);                \
  }                                                                        \
  if( CUDA_SUCCESS != cuDeviceGet( &dev, 0 ) ) {                        \
    fprintf(stderr, "CUDA: Cannot get the device\n"); exit(-1);                \
  }                                                                        \
  if( CUDA_SUCCESS != cuCtxCreate( &context, 0, dev ) ) {                \
    fprintf(stderr, "CUDA: Cannot create the context\n"); exit(-1);        \
  }                                                                        \
  if( CUBLAS_STATUS_SUCCESS != cublasInit( ) ) {                        \
    fprintf(stderr, "CUBLAS: Not initialized\n"); exit(-1);                \
  }                                                                        \
  //printout_devices( );
}

void finalizeGpu()
{
  cuCtxDetach( context );                        \
  cublasShutdown();
}

///////////////////////////////////////////////////////////////////////////////
float *calculateInverseSquareRootGpu(float *A, int N, float *W, float *Z)
{
  magma_timestr_t magmaStart, magmaTotalStart;
  cublasStatus_t status;
  float *h_work, *w;
  magma_int_t *iwork;
  
  magma_int_t info;

  char *uplo = (char*)MagmaUpperStr;
  char *jobz = (char*)MagmaVectorsStr;

  w=(float *)malloc(N*sizeof(float));

  magma_int_t nb = magma_get_dsytrd_nb(N);
  magma_int_t lwork = N*nb + 6*N + 2*N*N; //1 + 6*N*nb + 2*N*N;
  magma_int_t liwork = 3 + 5*N;

  CUDA_CHECK(cudaMallocHost( (void**)&h_work, lwork*sizeof(float) ));
  iwork=(magma_int_t *)malloc(liwork*sizeof(magma_int_t));
  
  magmaStart = get_current_time();
  
  // Note that the result is discarded, the CPU results are used
  magma_ssyevd(jobz[0], uplo[0],
               N, A, N, w,
               h_work, lwork,
               iwork, liwork,
               &info);
  
  printf("Eigendecomposition magma_ssyevd:\t%f\n",  GetTimerValue(magmaStart,get_current_time())/1000.);
  magmaTotalStart = get_current_time();

  float *d_A;
  CUDA_CHECK(cudaMalloc( (void**)&d_A, (N*N)*sizeof(float) ));
  // Note that the eigenvectors are discarded, the CPU results are used
  cublasSetMatrix( N, N, sizeof( float ), Z, N, d_A, N );
  status=cublasGetError();
  if (CUBLAS_STATUS_SUCCESS !=  status){
    printf("CUBLAS problem: %d\n", status);
  }

  /* The Gpu-resident version does not appear to work correctly
  float *h_worka;
  CUDA_CHECK(cudaMallocHost( (void**)&h_worka, N*N*sizeof(float) ));
  start = clock();
  magma_ssyevd_gpu(jobz[0], uplo[0],
               N, d_A, N, w,
               h_worka, N,
               h_work, lwork,
               iwork, liwork,
               &info);
  printf("Eigendecomposition magma_ssyevd_gpu:\t%f\n",  getTimerValue(start));
  cudaFreeHost(h_worka);
  */

  magmaStart = get_current_time();

  // Deal with the odd negative eigenvalues  
  for (int j=N-1; j>=0; --j) {
    if (w[j]<0){
      w[j]=w[j+1];
    }
  }
  
  // Allocate and initialise a new matrix B=Z*D
  // Note that the eigenvectors are discarded, the CPU results are used
  float *B = (float *)malloc(N*N*sizeof(float));
  for (int j=0; j<N; ++j) {
    float  lambda=sqrt(W[j]);
    if (lambda!=0.0){
      lambda=1.0/lambda;
    }else{
      printf("Zero eigenvalue!\n");
    }
    for (int i=0; i<N; ++i) {
      set_entry(B, N, i, j, get_entry(Z, N, i, j)*lambda);
    }
  }

  // Calculate the square root C=B*Z^T*/
  float *d_B, *d_C;
  CUDA_CHECK(cudaMalloc( (void**)&d_B, (N*N)*sizeof(float) ));
  CUDA_CHECK(cudaMalloc( (void**)&d_C, (N*N)*sizeof(float) ));

  cublasSetMatrix( N, N, sizeof( float ), B, N, d_B, N );
  status=cublasGetError();
  if (CUBLAS_STATUS_SUCCESS !=  status){
    printf("CUBLAS problem: %d\n", status);
  }

  cudaMemset((void*)d_C, 0, sizeof(float)*N*N);
/*
  cublasSgemm( 'N', 'T', N, N, N, 
                       1, d_B, N,
                          d_A, N,
                       0, d_C, N );
 */

  magmablas_sgemm( MagmaNoTrans, MagmaTrans, N, N, N, 
                         1, d_B, N, 
                            d_A, N,
                         0, d_C, N );

  status = cublasGetError();
  if (CUBLAS_STATUS_SUCCESS !=  status){
    printf("CUBLAS problem: %d\n", status);
  }

  printf("Square root:\t%f\n",  GetTimerValue(magmaStart,get_current_time())/1000.);
  printf("Square root with transfer:\t%f\n",  GetTimerValue(magmaTotalStart,get_current_time())/1000.);
  cudaFree( d_B );
  cudaFree( d_A );
  free(B);
  free(iwork);
  cudaFreeHost(h_work);
  free(w);
  
  return d_C;
}

float *transformOperatorGpu(float *O, float *d_NInverseSquareRoot, int nRows)
{
  magma_timestr_t magmaStart, magmaTotalStart;
  cublasStatus_t status;
  float *d_O, *d_OTemp;
  CUDA_CHECK(cudaMalloc( (void**)&d_O, (nRows*nRows)*sizeof(float) ));
  CUDA_CHECK(cudaMalloc( (void**)&d_OTemp, (nRows*nRows)*sizeof(float) ));

  magmaTotalStart = get_current_time();
  cublasSetMatrix( nRows, nRows, sizeof( float ), O, nRows, d_O, nRows );
  cudaMemset((void*)d_OTemp, 0, sizeof(float)*nRows*nRows);
  
  magmaStart = get_current_time();
/*
  cublasSsymm('L', 'U', nRows, nRows, 
              1, d_NInverseSquareRoot, nRows, d_O, nRows, 0, d_OTemp, nRows);
  status = cublasGetError();
  if (CUBLAS_STATUS_SUCCESS !=  status){
    printf("CUBLAS problem: %d\n", status);
  }
*/
  
  magmablas_sgemm( MagmaNoTrans, MagmaNoTrans, nRows, nRows, nRows, 
                         1, d_NInverseSquareRoot, nRows, 
                            d_O, nRows,
                        0, d_OTemp, nRows );
  
  magmablas_sgemm( MagmaNoTrans, MagmaNoTrans, nRows, nRows, nRows, 
                         1, d_OTemp, nRows,
                            d_NInverseSquareRoot, nRows,
                         0, d_O, nRows );
  printf("Basis transformation of operator:\t%f\n",  GetTimerValue(magmaStart,get_current_time())/1000.);
  printf("Basis transformation of operator with transfer:\t%f\n",  GetTimerValue(magmaTotalStart,get_current_time())/1000.);
  
  cudaFree( d_NInverseSquareRoot );
  cudaFree( d_OTemp );
  return d_O;
}
