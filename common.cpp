#include <cstdio>
#include <ctime>
#include <sstream>
#include <iostream>
#include <fstream>

#include "dqc.h"

using namespace std;

// Matrices are in Fortran style
void set_entry(float *A, int nRows, int i, int j, float val)
{
  A[j*nRows+i] = val;
}

float get_entry(const float *A, int nRows, int i, int j)
{
  return A[j*nRows+i];
}

///////////////////////////////////////////////////////////////////////////////
void printMatrix(float *A, int nRows, int nCols){
  for (int i=0; i<nRows; ++i) {
    for (int j=0; j<nCols; ++j) {
      float  x = get_entry(A, nRows, i, j);
      printf("%4.3f ", x);
    }
    putchar('\n');
  }
  putchar('\n');

}

///////////////////////////////////////////////////////////////////////////////
float *read_matrix(const char *inFileName, int &nRows, int &nCols){
  ifstream file;
  file.open(inFileName);
  string line;
  int elements = 0;
  while(getline(file,line))
  {
    stringstream linestream(line);
    string value;
    while(getline(linestream,value,' '))
    {
      if(value.length()>0){
        elements++;
      }
    }
    if (nRows==0){
      nCols=elements;
    }
    nRows++;
  }
  float *data=new float[elements];
  file.close();file.open(inFileName);
  int i=0, j=0;
  float data_element=0;
  while(getline(file,line))
  {
    stringstream linestream(line);
    string value;
    while(getline(linestream,value,' '))
    {
      if (value.length()>0){
        istringstream myStream(value);
        myStream >> data_element;
        set_entry(data, nRows, i,j,data_element);
        j++;
      }
    }
    j=0;
    i++;
  }
  file.close();
  return data;
}

float getTimerValue(clock_t start)
{
  return ( std::clock() - start ) / (float)CLOCKS_PER_SEC;
}
