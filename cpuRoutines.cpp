#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

extern "C" {
#include <cblas.h>
}

#include "dqc.h"

///////////////////////////////////////////////////////////////////////////////
extern "C" {
  static int ssyevr(char JOBZ, char RANGE, char UPLO, int N,
		   float *A, int LDA, float VL, float VU,
		   int IL, int IU, float ABSTOL, int *M,
		   float *W, float *Z, int LDZ, int *ISUPPZ,
		   float *WORK, int LWORK, int *IWORK, int LIWORK)
	{
	  extern void ssyevr_(char *JOBZp, char *RANGEp, char *UPLOp, int *Np,
						  float *A, int *LDAp, float *VLp, float *VUp,
						  int *ILp, int *IUp, float *ABSTOLp, int *Mp,
						  float *W, float *Z, int *LDZp, int *ISUPPZ,
						  float *WORK, int *LWORKp, int *IWORK, int *LIWORKp,
						  int *INFOp);
	  int INFO;
	  ssyevr_(&JOBZ, &RANGE, &UPLO, &N, A, &LDA, &VL, &VU,
			  &IL, &IU, &ABSTOL, M, W, Z, &LDZ, ISUPPZ,
			  WORK, &LWORK, IWORK, &LIWORK, &INFO);
	  return INFO;
	}
}

///////////////////////////////////////////////////////////////////////////////
extern "C" {
  static float slamch(char CMACH)
	{
	  extern float slamch_(char *CMACHp);
	  return slamch_(&CMACH);
	}
}

//////////////////////////////////////////////////////////////////////////////
float *createGramMatrix(float *A, int nRows, int nCols, float sigma)
{
    // Calculate the Gram matrix N
  float *N = (float *)malloc(nRows*nRows*sizeof(float));
  
  int i;
  clock_t start=clock();
  #pragma omp parallel shared(A,N) private(i)
  {
    #pragma omp for nowait
    for (i=0;i<nRows;i++){
      for (int j=0;j<nRows;j++){
        float sum=0.0;
        for (int k=0;k<nCols;k++){
          float a=get_entry(A, nRows, i,k);
          float b=get_entry(A, nRows, j,k);
          sum+=(a-b)*(a-b);
        }
        sum=sum/(4.0*sigma*sigma);
        sum=exp(-sum);
        set_entry(N, nRows, i, j, sum);
      }
    }
  }
  printf("Gram matrix: %f\n", getTimerValue(start));
  printf("Gram matrix with transfer: %f\n", getTimerValue(start));
  return N;
}

///////////////////////////////////////////////////////////////////////////////
float *createHamiltonian(float *A, float *N, int nRows, int nCols, 
                             float sigma, float mass)
{
  float *H = (float *)malloc(nRows*nRows*sizeof(float));
  
  // Kinetic part
  // Helper arrays for potential part
  float *psi = (float *)malloc(nRows*nRows*sizeof(float));
  float *psixixj = (float *)malloc(nRows*nRows*sizeof(float));

  int i;
  clock_t start=clock();
  #pragma omp parallel shared(A, H, psi, psixixj) private(i)
  {
    #pragma omp for nowait
    for (i=0;i<nRows;i++){
      for (int j=0;j<nRows;j++){
        float sum=0.0;
        for (int k=0;k<nCols;k++){
          float a=get_entry(A, nRows, i,k);
          float b=get_entry(A, nRows, j,k);
          sum+=(a-b)*(a-b);
        }
        sum=(1.0/(2.0*mass))*get_entry(N, nRows, i, j)/(2.0*sigma*sigma)*sum;
        set_entry(H, nRows, i, j, sum);
      }
    }
  
    #pragma omp for nowait
    for (i=0;i<nRows;i++){
      for (int j=0;j<nRows;j++){
        float sum=0.0;
        for (int k=0;k<nCols;k++){
          float a=get_entry(A, nRows, i,k);
          float b=get_entry(A, nRows, j,k);
          sum+=(a-b)*(a-b);
        }
        float psiElement=sum/(8.0*sigma*sigma);
        psiElement=exp(-psiElement);
        set_entry(psi, nRows, i, j, psiElement);
        set_entry(psixixj, nRows, i, j, psiElement*sum);
      }
    }

    // Potential part
    #pragma omp for nowait
    for (i=0;i<nRows;i++){
      for (int j=0;j<nRows;j++){
        float sumPsi=0.0;
        float sumPsixixj=0.0;
        for (int k=0;k<nRows;k++){
          sumPsi+=get_entry(psi, nRows, k, j);
          sumPsixixj+=get_entry(psixixj, nRows, k, j);
        }
        sumPsi*=8*sigma*sigma;
        float element=get_entry(N, nRows, i, j)*(-(float)nCols/2.0+(1.0/sumPsi)*sumPsixixj);
        element+=get_entry(H, nRows, i, j);

        set_entry(H, nRows, i, j, element);
      }
    }
  }
  free(psi);
  free(psixixj);
  printf("Hamiltonian: %f\n", getTimerValue(start));
  return H;
}

///////////////////////////////////////////////////////////////////////////////
float *calculateInverseSquareRoot(float *A, int N, float *W, float *Z)
{
  clock_t start;
  float *B, *WORK;
  int *ISUPPZ, *IWORK;
  int  M;
    
  // Allocate space for the output parameters and workspace arrays
//  W = (float *)malloc(N*sizeof(float));
//  Z = (float *)malloc(N*N*sizeof(float));
  ISUPPZ = (int *)malloc(2*N*sizeof(int));
  WORK = (float *)malloc(26*N*sizeof(float));
  IWORK = (int *)malloc(10*N*sizeof(int));

  start = clock();      
  // Get the eigenvalues and eigenvectors
  ssyevr('V', 'A', 'L', N, A, N, 0, 0, 0, 0, slamch('S'), &M,
         W, Z, N, ISUPPZ, WORK, 26*N, IWORK, 10*N);
  printf("Eigendecomposition:\t%f\n", getTimerValue(start));
  
    // Deal with the odd negative eigenvalues  
  for (int j=N-1; j>=0; --j) {
    if (W[j]<0){
      W[j]=W[j+1];
    }
  }

  start = clock();      
  // Allocate and initialise a new matrix B=Z*D
  B = (float *)malloc(N*N*sizeof(float));
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

  // Calculate the square root C=B*Z^T
  float *C = (float *)malloc(N*N*sizeof(float));
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, N, N, N,
              1, B, N, Z, N, 0, C, N);
  printf("Square root:\t%f\n",  getTimerValue(start));               
  printf("Square root with transfer:\t%f\n",  getTimerValue(start)); 
  free(B);
//  free(W);
//  free(Z);
  free(WORK);
  free(ISUPPZ);
  free(IWORK);

  return C;
}

float *transformOperator(float *O, float *NInverseSquareRoot, int nRows){
  clock_t start;
  float *OTemp = (float *)malloc(nRows*nRows*sizeof(float));
  start = clock();                
  /*
  cblas_ssymm(CblasColMajor,  CblasLeft, CblasUpper, nRows, nRows, 
              1, NInverseSquareRoot, nRows, O, nRows, 0, OTemp, nRows);
              */
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nRows, nRows, nRows,
              1, NInverseSquareRoot, nRows, O, nRows, 0, OTemp, nRows);
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nRows, nRows, nRows,
              1, OTemp, nRows, NInverseSquareRoot, nRows, 0, O, nRows);
  printf("Basis transformation of operator:\t%f\n",  getTimerValue(start));
  printf("Basis transformation of operator with transfer:\t%f\n",  getTimerValue(start));
  free(OTemp);
  return O;
}
