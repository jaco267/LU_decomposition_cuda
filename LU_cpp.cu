#include <iostream>
#include "func.cuh"
#include <bits/stdc++.h>
#include "print_utils.cuh"
#include <chrono>
using std::cout;
using std::endl;
using namespace std;

const int MAX = 600;
void swap_rows(float* A, int n, int r1, int r2){
  if(r1 == r2) return;
  for(int j=0; j<n; j++){
      std::swap(A[r1*n + j], A[r2*n + j]);
  }
}
int luDecomposition(float* mat_in, int n,
  bool verbose= true, bool pivoting = false, bool check_equal = false
){
  // 使用 high_resolution_clock 獲得最高精度
  auto start = std::chrono::high_resolution_clock::now();
  //--------------------------------------------------
  float* mat = new float [n*n];
  vector<int> P(n);
  for(int i=0; i<n; i++) P[i] = i;
  for(int i=0; i<n ; i++){
    for(int j=0; j<n ; j++){mat[i*n+j] = mat_in[i*n+j];}
  }
  
  float* upper  = new float[n*n];
  float* lower  = new float[n*n]; 
  memset(lower, 0, n*n*sizeof(float)); 
  memset(upper, 0, n*n*sizeof(float));
  // for(int i=0; i<n*n; i++) cout<<mat[i]<<" ";   cout<<endl;
  
  // Decomposing matrix into Upper and Lower triangular matrix
  for (int i = 0; i < n; i++) {
    int pivot = i;
    float max_val = fabs(mat[i*n + i]);
    // if (i==2){
    //   cout<< "max_val "<<max_val<<endl;
    //   cout<<"mat\n"; print_mat2(mat,n,n);
    //   cout<<"upper\n"; print_mat2(upper,n,n);
    //   cout<<"lower\n"; print_mat2(lower, n,n);
    //   cout<<" P: ";     for(int k=0;k<n; k++) cout<<P[k]<<" ";   cout<<endl;
    //   print_mat2(mat, n,n);
    //   exit(-1);
    // }
    if (pivoting){
      for(int r = i+1; r < n; r++){
        float v = fabs(mat[r*n + i]);
        if(v > max_val){max_val = v;  pivot = r;}
      }
    }
    // cout<< "max_val "<<max_val<<endl;
    // cout<<"pivot: "<<pivot<<endl;


    if (pivoting){// 如 pivot row != i, 則交換
      if(pivot != i){
          swap_rows(mat, n, i, pivot);
          swap_rows(lower, n, i, pivot);   // 只交換 lower 的前 i 欄
          swap(P[i], P[pivot]);
      }
    }

    // Upper Triangular
    for (int k = i; k < n; k++){// Summation of L(i, j) * U(j, k)
      float sum = 0;
      for (int j = 0; j < i; j++) sum += (lower[i*n+j] * upper[j*n+k]);
      upper[i*n+k] = mat[i*n+k] - sum;// Evaluating U(i, k)
    }

    // Lower Triangular
    for (int k = i; k < n; k++) {
      if (i == k)  lower[i*n+i] = 1; // Diagonal as 1
      else {// Summation of L(k, j) * U(j, i)
        float sum = 0;
        for (int j = 0; j < i; j++)  sum += (lower[k*n+j] * upper[j*n+i]);
        // **檢查 / 0**
        if (std::abs(upper[i*n+i]) < 1e-9) { cout << "singular mat,need pivoting." << endl;exit(-1);}
        // Evaluating L(k, i)
        lower[k*n+i] = (mat[k*n+i] - sum) / upper[i*n+i];
      }
    }

  }
  //---------------------------------------
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  // setw is for displaying nicely
  if(verbose){
    cout << setw(6) << "      Lower Triangular" << setw(32)<< "Upper Triangular" << endl;
    for (int i = 0; i < n; i++) {
       for (int j = 0; j < n; j++) cout << setw(6) << lower[i*n+j] << "\t";cout << "\t";// Lower
       for (int j = 0; j < n; j++) cout << setw(6) << upper[i*n+j] << "\t";cout << endl; // Upper
    }
  }
  int all_equal = 0;
  if (check_equal){
    float* in  = new float[n*n]; float* out = new float[n*n]; //* use heap 避免  stack overflow (stack ~8mb, heap~= 8GB)
    float* out_final = new float[n*n];
    float* LL = new float[n*n];  float* UU = new float[n*n]; 
    for (int i = 0; i <n; i++) {
      for(int j =0; j<n; j++){
        LL[i*n+j] = lower[i*n+j]; UU[i*n+j] = upper[i*n+j]; in[i*n+j] = mat[i*n+j];
      }
    }
    mat_mul(LL,UU,out,n,n);
    if (verbose){
      cout<<"-----in---"<<endl;  print_mat2(mat_in,n,n);
      cout<<"-----out---"<<endl; print_mat2(out,n,n);
    }
    if (pivoting){
      for (int i = 0; i < n; i++){
          for (int j =0; j < n ; j++){
            // place A_rec[i] to the row where original row i should go
            out_final[ P[i]*n+j ] = out[i*n+j];
          }
      }
      if(verbose){cout<<"-----out final---"<<endl; print_mat2(out_final,n,n);}
    }
     
    if (pivoting){ all_equal = check_equal2(mat_in,out_final, n,n);
    }else{         all_equal = check_equal2(mat_in,out, n,n);}
    if (all_equal!=1){cout<<"not equal!!!!"<<endl;
    }else if (all_equal == 1){cout<<"-------all_equal: true, " <<all_equal<<endl;  }
    delete[] LL;  delete[] UU;
  }
  // 輸出時間
  std::cout << "LU Decomposition (CPU) Time: " << duration.count() 
     << " microseconds (" << duration.count() / 1000.0 << " ms)\n";
  delete[] mat;  delete[] upper;  delete[] lower;
  if (verbose){
    cout<<"---P---"<<endl; 
    for(int i=0; i<n; i++){cout<<P[i]<<" ";} cout<<endl;
  }
  return all_equal;
}

int run_LU_ex1(bool pivoting = false){
    cout<<"LU"<<endl;
    cout<<"---pivoting---"<<pivoting<<endl;
    // float mat[3*3] = {  2, 4, 8 , 
    //                     1, 3, 7 , 
    //                     1, 3, 9  };
    // float mat[3*3] = { 2, 2, 8, 
    //                    7, 7, 10, 
    //                    6, 8, 1  };     
    // float mat[3*3] = { 2, 7, 6, 
    //                    2, 7, 8, 
    //                    8, 10, 1  };     
    float mat[3*3] = { 9, 3, 5, 
                       4, 8, 5, 
                       7, 5, 0  }; 
    // exit(-1);    
    print_mat2(mat,3,3);         
    bool verbose = true ; 
    int all_equal = luDecomposition(mat, 3, 
      verbose, pivoting, true); 
    return all_equal;
}
int run_LU_ex2(int kern_size = 4, bool verbose = true,
   bool pivoting = false, bool check_equal = false){
  
  int size= kern_size;   

  float* mat_in = new float[size*size]; 
  
  for (int i = 0; i < size*size; i++) {
      mat_in[i] = (float)rand() / RAND_MAX;
      mat_in[i] = std::round(mat_in[i]*10);
  }
  if (verbose){
    cout<<"----mat---\n";
    print_mat2(mat_in, size,size);

  }
  // mat_in[0] = 1; mat_in[1] = 5; mat_in[2] = 2; mat_in[3] = 1;  
  // mat_in[4] = 9; mat_in[5] = 2; mat_in[6] = 6; mat_in[7] = 2;
  // mat_in[8] = 8; mat_in[9] = 5; mat_in[10]= 7; mat_in[11] = 3;
  // mat_in[12] = 2; mat_in[13] = 4; mat_in[14]= 8; mat_in[15] = 6;
  //     = { { 2, -1, -2 }, { -4, 6, 3 }, { -4, -2, 8 } };
  int all_equal = luDecomposition(mat_in, size,verbose,pivoting, check_equal);
   delete[] mat_in;
  return all_equal;
}