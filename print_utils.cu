#include "print_utils.cuh"

void print_mat(float* h_A, int M, int N){
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            std::cout << h_A[j*M + i] << " ";
        }
        std::cout << std::endl;
    }
}

void print_mat2(float* h_A, int M, int N){
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            std::cout << h_A[i*N+j] << " ";
        }
        std::cout << std::endl;
    }
}

void mat_mul(float* LL, float* UU, float* LLUU, int M, int N){
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            LLUU[i*N+j] = 0.0f;
            for (int k = 0; k < N; k++) {  // or min(M, N)
                LLUU[i*N+j] += LL[i*N+k] * UU[k*N+j];
            }
        }
    }
}
int check_equal(float* LLUU,float* in_mat, int M, int N){
  int all_equal = 1; 
  for(int i=0;i<M;i++){
    for(int j=0;j<N;j++){
      if(fabs( LLUU[j*M + i]-in_mat[i*N + j])>0.001){
          // cout<<"errr not equal"<<endl;
          all_equal = 0;
      } 
    }
  }
  return all_equal;
}
int check_equal2(float* LLUU,float* in_mat, int M, int N){
  int all_equal = 1; 
  for(int i=0;i<M;i++){
    for(int j=0;j<N;j++){
      if(fabs( LLUU[i*N + j]-in_mat[i*N + j])>0.001){
        // cout<<"errr not equal: i="<<i<<", j="<<j<<endl;
        // cout<<"val0="<<LLUU[i*N + j]<<", val1="<<in_mat[i*N+j]<<endl;
        all_equal = 0;
      } 
    }
  }
  return all_equal;
}