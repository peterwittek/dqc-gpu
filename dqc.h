void set_entry(float *A, int nRows, int i, int j, float val);
float get_entry(const float *A, int nRows, int i, int j);
void printMatrix(float *A, int nRows, int nCols);
float *read_matrix(const char *inFileName, int &nRows, int &nCols);
float getTimerValue(clock_t start);

float *createGramMatrix(float *A, int nRows, int nCols, float sigma);
float *createHamiltonian(float *A, float *N, int nRows, int nCols, 
                             float sigma, float mass);
float *calculateInverseSquareRoot(float *A, int N, float *W, float *Z);
float *transformOperator(float *O, float *NInverseSquareRoot, int nRows);

void initializeGpu();
void finalizeGpu();
float *calculateInverseSquareRootGpu(float *A, int N, float *W, float *Z);
float *transformOperatorGpu(float *O, float *NInverseSquareRoot, int nRows);

extern "C" {
float *createGramMatrixGpu(float *A, int nRows, int nCols, float sigma);
}
