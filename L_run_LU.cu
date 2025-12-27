#include <iostream>
#include "func.cuh"
#include <cublas.h>
#include "print_utils.cuh"
#include <cuda_runtime.h>
using std::cout;
using std::endl;
void pivot_steps_to_perm(int *P, int M, int *perm_out) {
    // start with identity
    for (int i = 0; i < M; ++i) perm_out[i] = i;
    int minDim = M; // or min(M,N) if rectangular; but safe to use M and ignore tail
    for (int k = 0; k < minDim; ++k) {
        int piv = P[k];
        // swap perm_out[k] and perm_out[piv]
        int tmp = perm_out[k];
        perm_out[k] = perm_out[piv];
        perm_out[piv] = tmp;
    }
}
void run_cuda_LU(bool pivoting = false){

    int device;
    cudaDeviceProp props;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);
    printf("Maximum threads per block: %d\n", props.maxThreadsPerBlock); //* 1024 ,  so in v1 if N>1024 will have error 
    printf("Max block dimensions: x=%d, y=%d, z=%d\n",
        props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
    //* thread per block xyz 1024,2024,64
    printf("Max grid dimensions: x=%d, y=%d, z=%d\n",
           props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);
    //* block per grid 2147483647,65535,65535
    // cout<<"-------"<<endl;

    int M, N, lda;

    M = 3; N = 3; lda = 3;
    // float h_A[M*N] = { 2, 1, 1, 
    //                  4, 3, 3, 
    //                  8, 7, 9 }; // column-major
    float h_A[M*N] = { 0, 2, 1, 
                     2, 7, 3, 
                     8, 5, 9 }; // column-major
    print_mat(h_A, 3,3);
    // M = 4; N = 2; lda = 4;
    // float h_A[M*N] = { 8, 2, 7, 4, 
    //                    3, 9, 4, 5 }; // column-major

    int P[M];
    for(int i=0; i<M;i++){ P[i] = i;}
    InfoStat stat;
    float* d_A;
    cudaMalloc(&d_A, sizeof(float)*M*N);
    cudaMemcpy(d_A, h_A, sizeof(float)*M*N, cudaMemcpyHostToDevice);
    // cublasHandle_t handle;
    // cublasCreate(&handle);
    DecomposeLU(M, N, lda, d_A, P, 1e-6f, stat, pivoting); // pass handle inside if using v2 API
    cudaMemcpy(h_A, d_A, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
    // cublasDestroy(handle);
    cudaFree(d_A);
  
    // Print results
    // std::cout << "LU matrix:" << std::endl;
    // print_mat(h_A, M,N);



    float* LL = new float[M*N];
    float* UU = new float[M*N];  
    getLU(LL,UU,h_A,M,N,lda);
    float* LLUU= new float[M*N]; //todo  seg fault in here 
    cout << "----LU = L*U---!\n"; 
    mat_mul(LL,  UU, LLUU, M, N);
    print_mat2(LLUU,M,N);
    cout<<"----L---"<<endl;
    print_mat2(LL,M,N);
    cout<<"----U---"<<endl;
    print_mat2(UU,M,N);
    //* p = .......
    cout<<"------p------"<<endl;
    // for(int i=0; i<M; i++){  cout<<P[i]<<" ";} cout<<endl;
    
    int Perm_out[M];
    pivot_steps_to_perm(P, M, Perm_out);
    for(int i=0; i<M; i++){  cout<<Perm_out[i]<<" ";} cout<<endl;

    /*
    ex. input                                L          U
    2, 1, 1, transpose  2 4 8  LU decomp    1   0  0 | 2 4 8 
    4, 3, 3, -------->  2 3 7  --------->   0.5 1  0 | 0 1 3 
    8, 7, 9             1 3 9               0.5 1  1 | 0 0 2  
        compress ver
    ->   2   4 8 
         0.5 1 3
         0.5 1 2 
    */   
}
int run_block_LU_cuda(float* h_A, int M, int N, int lda,
     bool verbose = true, bool pivoting = false, bool check_equal_flag = false){
  float* in_mat = new float[M*N];
  memcpy(in_mat,h_A, M*N*sizeof(float)); 
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  int P[M];
  for(int i=0; i<M;i++){ P[i] = i;}
  InfoStat stat;
  float* d_A;
  cudaMalloc(&d_A, sizeof(float)*M*N);
  cudaMemcpy(d_A, h_A, sizeof(float)*M*N, cudaMemcpyHostToDevice);
  int block_num = 100;  //* block_num = 2,3 will have incorrect result   
  // cublasHandle_t handle; cublasCreate(&handle);
  // DecomposeLU(M, N, lda, d_A, P,  1e-6f, stat);
  cudaEventRecord(start, 0);
  DecomposeBlockedLU(M, N, lda, d_A, P, block_num, 1e-6f, stat,pivoting); // pass handle inside if using v2 API
  cudaEventRecord(stop, 0);
  cudaMemcpy(h_A, d_A, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
  // cublasDestroy(handle);
  cudaFree(d_A);
  
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  int all_equal; 
  if (check_equal_flag){
    float* LL = new float[M*N]; float* UU = new float[M*N];  
    getLU(LL,UU,h_A,M,N,lda);
    int Perm_out[M];
    pivot_steps_to_perm(P, M, Perm_out);
    if(verbose){
        cout<<"----pivot---"<<endl;
        for(int i=0; i<M; i++){  cout<<Perm_out[i]<<" ";} cout<<endl;
    } 
    float* LLUU= new float[M*N]; //todo  seg fault in here 
    float* LU_final = new float[M*N];    
    mat_mul(LL,  UU, LLUU, M, N);
    for (int i = 0; i < M; i++){
        for (int j =0; j < N ; j++){
          // place A_rec[i] to row where original row i should go
          LU_final[ Perm_out[i]*N+j ] = LLUU[i*N+j];
        }
    }
    if (verbose){
        if (pivoting){cout<<"----LU final ---!\n"; print_mat2(LU_final, M,N); }
        cout << "----LU = L*U---!\n"; print_mat2(LLUU,M,N);
        cout<<"----L---"<<endl;       print_mat2(LL,M,N);
        cout<<"----U---"<<endl;       print_mat2(UU,M,N);
    }   
    if (pivoting){all_equal = check_equal(LU_final,in_mat,M,N);
    }else{all_equal = check_equal(LLUU,in_mat,M,N);}
    if (all_equal!=1){        cout<<"not equal!!!!"<<endl;
    }else if (all_equal == 1){cout<<"all_eqaual: true, " <<all_equal<<endl;  }
    delete[] LLUU; delete[] LL;  delete[] UU; 
  }
  delete[] in_mat; 
  std::cout << "LU Decomposition Time: " << milliseconds << " ms\n";
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return all_equal;
}
void run_cuda_block_LU_ex1(bool pivoting = false, bool check_equal_flag = false){
    // int size  = 4;  //*size = 400 will neq  
    // int M = size, N = size, lda = size;
    // float h_A[M*N] = { 8, 2, 7, 4, 
    //                   3, 9, 4, 5, 
    //                   1, 5, 2, 3,
    //                   6, 7, 4, 3 }; // column-major    
    int size  = 3;  //*size = 400 will neq  
    int M = size, N = size, lda = size;
    // float h_A[M*N] = { 2, 2, 8, 
    //                    7, 7, 10, 
    //                    6, 8, 1}; // column-major  
    // float h_A[M*N] = { 2, 7, 6, 
    //                    2, 7, 8, 
    //                    8, 10, 1}; // column-major  
    // float h_A[M*N] = { 9, 3, 5, 
    //                    4, 8, 5, 
    //                    7, 5, 0}; // column-major  
    float h_A[M*N] = { 9, 4, 7, 
                       3, 8, 5, 
                       5, 5, 0}; // column-major  
    print_mat(h_A,M,N);
    bool verbose = true;
    int all_equal = run_block_LU_cuda(h_A,M,N,lda,
                                     verbose,pivoting,check_equal_flag);
    cout<<"---final ha---\n";
    print_mat(h_A,M,N);
}
  
int run_cuda_block_LU_ex2(int kern_size = 4,bool verbose = true, 
    bool pivoting = false, bool check_equal = false
){
    int size  = kern_size;  //*size = 400 will neq
    //*int size = 4;  
    int M = size, N = size, lda = size;
    // float h_A[M*N] = { 8, 2, 7, 4, 
    //                   3, 9, 4, 5, 
    //                   1, 5, 2, 3,
    //                   6, 7, 4, 3 }; // column-major   
    // float h_A[M*N];
    float* h_A = new float[M*N]; 
    for (int i = 0; i < M * N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_A[i] = std::round(h_A[i]*10);
    }
    
    if (verbose){
        cout<<"input matrix\n";
        print_mat2(h_A,M,N);
    }
    int all_equal = run_block_LU_cuda(h_A,M,N,lda,verbose,pivoting,check_equal);
    delete [] h_A;
    return all_equal;
}
  
  
  