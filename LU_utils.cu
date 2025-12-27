#include <iostream>
#include "func.cuh"
#include <cublas.h>
using std::cout;
using std::endl;

void getLU(float* L, float* U, float* h_A, int M, int N,int lda){
  for(int i =0; i<M; i++){
    for(int j=0; j<N; j++){
      if (i==j){      L[i*N+j]=1;
      }else if (j<i){ L[i*N+j] = h_A[j*lda+i]; 
      }else{          L[i*N+j]= 0;}
    }
  }

  for(int i =0; i<M; i++){
    for(int j=0; j<N; j++){
      if (i<=j){      U[i*N+j] = h_A[j*lda+i]; 
      }else{          U[i*N+j]= 0;}
    }
  }
}