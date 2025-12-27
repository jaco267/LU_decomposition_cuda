#pragma once

struct InfoStat { //todo ????
    int _info;
    std::string _str;
  };

void run_vec_add();
void DecomposeLU(int M, int N, int lda , float* A, int* P, 
  float epsilon, InfoStat& stat, bool pivoting);

void DecomposeBlockedLU (int M, //  Num of rows of A  
  int N, // Num of column of A  
  int lda,  //Leading dim of A lda < std::max(1,M) 
  float *A,  // Float Matrix of size M*N on the output contains the result of the LU decomposition  
             // The diagonal elements for L are not stored in A ( assuming they are all 1)  
  int* P, // Permutation vector of size M
  int blockSize, //Size of the submatrices, if blockSize>=M || blockSize==1 unblocked decomposition is called  
  float epsilon, // Epsilon (used to test for singularities)   
  InfoStat &stat, // return status 
  bool pivoting
); 
void run_cuda_LU(bool pivoting);
int run_LU_ex1(bool pivoting);
int run_LU_ex2(int kern_size, bool verbose, bool pivoting,
  bool check_equal);
void run_cuda_block_LU_ex1(bool pivoting, bool check_equal_flag);
int run_cuda_block_LU_ex2(int kern_size,bool verbose, 
  bool pivoting, bool check_equal);
//* LU_utils.cu 
void getLU(float* L, float* U, float* h_A, int M, int N,int lda);