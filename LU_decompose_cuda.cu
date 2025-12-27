#include <iostream>
#include "func.cuh"
#include <cublas.h>
#include <vector>
using std::cout;
using std::endl;
// **************************************************************************************  
//! this is just standard LU decomposition  
void DecomposeLU(int M, // Num of rows of A  
    int N,              // Num of column of A   
    int lda , //  Leading dim of A lda < std::max(1,M)  
    float* A, //  Float Matrix of size M*N on the output contains the result of the LU decomposition  
              //  The diagonal elements for L are not stored in A ( assuming they are all 1)     
    int* P,   //   Permutation vector of size M  
    float epsilon, //Epsilon (used to test for singularities)  
    InfoStat& stat, // return status
    bool pivoting = false 
)  {  
    /*float h_A[M*N] = { 8, 2, 7, 4, 
                         3, 9, 4, 5 };
    [8  3] pivot = 8    [8     3]  
    [2  9]  row[1:]/8   [0.25  9]   9-0.25*3=8.25
    [7  4]  become ->   [0.875 4]   4-0.875*3 = 1.375
    [4  5]              [0.5   5]   5-0.5*3 = 3.5
      cublasSscal(...);
    should be  
    8    3 
    0.25 8.25
    0.875 0.1667
    0.5   0.4242   
    */
    cublasStatus_t cuStat;  
    //Preconditions  
    if ( M<=0 || N<=0 || lda < std::max(1,M) )       {  
         stat._info = -1;  
         if (M<=0)  stat._str = "M<=0";  
         if (N<=0)  stat._str = "M<=0";  
         if (lda < std::max(1,M))  stat._str = "lda < std::max(1,M)";  
         return;  
    }  
    
    int minDim = std::min( M, N );  //* 2  
    for (int k=0; k<minDim-1; k++) {   //minDim=3 k
        //  pivot Row=0=0-1+            (4  ,A+0+0, 1)
        //*  ???                            A[k][k]
        int idx1based = cublasIsamax(M - k, A + k + k*lda, 1);
        int pivotRow  = k + (idx1based - 1);  
        // int pivotRow = k-1+cublasIsamax(M-k,A+k + k*lda, 1); // row relative to the current submatrix 
        
        // cout<<"k:"<<k<<" pivotRow:"<<pivotRow<<" idx1based "<<idx1based<<" M: "<<M<<endl;
        
        
        int kp1 = k+1;  
        P[k] = pivotRow;  
        if (pivoting){
          if (pivotRow!=k){  cublasSswap(N, A+pivotRow, lda, A+k, lda);  }  
        }
        // if(k==1) return;
        float valcheck;  
        //* pivot node : valcheck= A[k][k]
        cublasGetVector(1,sizeof(float),A+k+ k*lda, 1, &valcheck, 1);  
        // cout<<"val_check:"<<valcheck<<endl;
        
        
        if (fabs(valcheck) < epsilon){  
          stat._info =k+1;cout<<"matrix is singular\n";  
          stat._str = " Matrix is Singular ";  return;  }  
        if (kp1 < M)  {  // row[1:] / valcheck=pivot_val=8
           //* A[kp1:M-kp1][k] = A[kp1:][k] *= 1/valcheck
            cublasSscal(M-kp1, 1.0f/valcheck,A+kp1+ k*lda, 1);  
        }  
        
        if ( kp1 < minDim ){  // A[kp1:][kp1:] -= A[kp1:M-kp1][k]@ A[k][kp1:N-kp1]
            //* A[kp1:][kp1:] -= A[kp1:][k] @ A[k][kp1:]
             cublasSger (M-kp1, N-kp1, -1.0f,A+kp1+ k*lda, 1, A+k+ kp1*lda, lda,A+ kp1*lda+kp1, lda);  
        }   
        if(((M-1-kp1)>0) && (kp1>=minDim-1)){//* newly added to fix the bug for 4,2
           cublasGetVector(1,sizeof(float),A+kp1+ kp1*lda, 1, &valcheck, 1);  
           cublasSscal(M-kp1-1, 1.0f/valcheck,A+kp1+1+ kp1*lda, 1);
        } 
    }  
}  


//! this is the block parallel version
void DecomposeBlockedLU (int M, //  Num of rows of A  
    int N, // Num of column of A  
    int lda,  //Leading dim of A lda < std::max(1,M) 
    float *A,  // Float Matrix of size M*N on the output contains the result of the LU decomposition  
               // The diagonal elements for L are not stored in A ( assuming they are all 1)  
    int* P, // Permutation vector of size M
    int blockSize, //Size of the submatrices, if blockSize>=M || blockSize==1 unblocked decomposition is called  
    float epsilon, // Epsilon (used to test for singularities)   
    InfoStat &stat, // return status
    bool pivoting = false 
)  {  
  cublasStatus cuStat;  
  //Preconditions  
  if (M < 0 || N < 0 || lda < std::max(1,M) )  {  
    stat._info = -1;  
    if (M<=0)   stat._str = "M<=0";  
    if (N<=0)   stat._str = "M<=0";  
    if (lda < std::max(1,M))  stat._str = "lda < std::max(1,M)";     
    return;     
  }   
  int minSize = std::min(M,N);    
  // cout<<"block size: "<<blockSize<<" min_size: "<<minSize<<endl; //4
  if ( blockSize > minSize || blockSize == 1)  {   
    //straight LU decomposition  
    // for(int i=0; i<M; i++){  cout<<P[i]<<" ";}cout<<endl;
    DecomposeLU( M, N, lda, A, P, epsilon, stat, pivoting);  
  }  else  {  
    //blocked decomposition  
    for (int i =0; i< minSize ; i+=blockSize)  {    
      int realBlockSize  = std::min(minSize - i, blockSize); //realBlockSize=min (4-0, 2) = 2  
      //decompose the current rectangular block  
      //             4,          2          A
      DecomposeLU( M-i, realBlockSize, lda, A+i+i*lda, P+i, epsilon, stat,pivoting);  
      //adjust pivot infos  
      //Todo : write a kernel for that  
      if(pivoting){
        for (int p = i; p< std::min( M, i+realBlockSize)-1; p++)  {  
          P[p] = P[p]+i;  
          if (P[p] != p)  {  
            cublasSswap(i, A+p , lda, A+ P[p], lda);// Apply interchanges to columns 0:i.  
            // Apply interchanges to columns i+blockSize:N.  
            cublasSswap(N-i-realBlockSize, A+p+(i+realBlockSize)*lda , 
            lda, A+ P[p]+(i+realBlockSize)*lda, lda);  
          }  
        }  
      }
      // Compute block row of U.  //??? matrix inverse???  U01=L00^{-1}A01
      /* cublasStrsm() is a cuBLAS function that solves a triangular linear system of equations, using single-precision (float) data*/
      //*                               M                N   
      cublasStrsm( 'l','l','n','u', realBlockSize, N-i-realBlockSize, 1.0f,  
           A +i +i*lda, lda, A +i + (i+realBlockSize)*lda, lda);  
          //* A=A[i:i+realBlockSize][i:i+realBlockSize], also its lower triangle   L00
                                                        // side = 'l',   AX = B    X = A^{-1}B
          //* B=A[i:i+realBlockSize][i+realBlockSize: ]               A:(M,M) , B (M,N)
          // CHECK_CUBLAS("decomposeBlockedLU cublasStrsm");  
      
      if (i+realBlockSize < M)  {  
        //* obtain A11'   
        cublasSgemm('n','n',  M-i-realBlockSize, N-i-realBlockSize, realBlockSize,  -1.0f,  
                 A+i+realBlockSize+i*lda,lda,  //L10 = A[i+realBlocksize:][i:i+realBlockSize]  
                 A+i+(realBlockSize+i)*lda,lda,//U01 = A[i:i+realBlockSize][i+realBlockSize:]  
                 1.0f,  
                 A+i+realBlockSize+(realBlockSize+i)*lda,lda //A11=A[i+realBlockSize:][i+realBlockSize:] 
        );  
        //  A'11 = A11-L10  U01   
        // CHECK_CUBLAS("decomposeBlockedLU cublasSgemm");  
      }  
    }  

  }  
}  
  

