#include <cstdlib>
#include <iostream>

#include "dqc.h"

//////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv )
{
  float sigma=0.1;
  float mass=sigma*sigma;

  if (argc!=2){
    std::cerr << "Wrong number of parameters!\n";
    exit(-1);
  }

  const char *inFileName = argv[1];
  int nRows=0, nCols=0;
  float *A = read_matrix(inFileName, nRows, nCols);
  
  float *N = createGramMatrix(A, nRows, nCols, sigma);
  //float *H = createHamiltonian(A, N, nRows, nCols, sigma, mass);
  float *H = N;

  // Eigenvalues and eigenvectors
  float *W = (float *)malloc(nRows*sizeof(float));
  float *Z=(float *)malloc(nRows*nRows*sizeof(float));

  float *NInverseSquareRoot=calculateInverseSquareRoot(N, nRows, W, Z);
  transformOperator(H, NInverseSquareRoot, nRows);
  //printMatrix(H, nRows, nRows);

  std::cout << "GPU calculations\n";
  initializeGpu();
  createGramMatrixGpu(A, nRows, nCols, sigma);

  //Note that N will contain the eigenvectors after the call
  float *d_NInverseSquareRoot = calculateInverseSquareRootGpu(N, nRows, W, Z);
  float *d_Htr = transformOperatorGpu(H, d_NInverseSquareRoot, nRows);

  finalizeGpu();
  free(W);
  free(Z);
//  free(H);
  free(A);
  free(N);
  return 0;
}

