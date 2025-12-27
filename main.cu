#include <iostream>
#include "func.cuh"
#include <string>
#include <cublas.h>
#include <bits/getopt_core.h>
using std::cout;
using std::endl;
using std::stoi;
/*
a thread block contina up to 1024 threads    
we need more blocks to execute more threads    
blocks are organized into grids   

thread         ---executed by--->  core  
thread block   ---executed by--->  streaming multiprocessor  
kernel grid    ---executed by--->  complete gpu unit

the thread ID can be accessed using  
int i = blockIdx.x * blockDim.x + threadIdx.x     
*/

int main(int argc, char *argv[]) {
  int opt; 
  int run_opt = 0;
  int kern_size = 4; 
  bool verbose = true;
  bool pivoting = false;
  bool check_equal = false;
  int mc_iter = 1;
  while((opt = getopt(argc, argv, ":i:o:s:v:p:e:m:x")) != -1) { 
    switch(opt) { 
      case 'o':  run_opt = stoi(optarg);  break; 
      case 's':  kern_size = stoi(optarg);  break; 
      case 'v':  verbose = stoi(optarg);  break;
      case 'p':  pivoting = stoi(optarg);  break; 
      case 'e':  check_equal = stoi(optarg);  break;
      case 'm':  mc_iter = stoi(optarg);  break;
      case ':': printf("option needs a value\n"); break; 
      case '?': printf("unknown option: %c\n", optopt); break; 
    } 
  } 
  cout<<"mc_iter: "<<mc_iter<<endl;
  for(; optind < argc; optind++){	 
      printf("extra arguments: %s\n", argv[optind]); 
  } 
  cout<<"option: "<<run_opt<<endl;
  int all_equal= 0; 
  if (run_opt == 0){
    run_cuda_LU(pivoting);
  }else if (run_opt==1){
    run_cuda_block_LU_ex1(pivoting,(bool)check_equal);
  }else if (run_opt==2){

    for(int i= 0; i<mc_iter; i++){
      all_equal =  run_cuda_block_LU_ex2(kern_size,verbose,pivoting,(bool)check_equal);
      if ((all_equal != 1) && check_equal){
        cout<<"warning::: not equal"<<endl;
        exit(-1);
      }
    }
  }else if (run_opt==-1){
    all_equal = run_LU_ex1(pivoting);
    if ((all_equal != 1) && check_equal){
      cout<<"warning::: not equal"<<endl;
      exit(-1);
    }
  }else if (run_opt==-2){
    cout<<"LU ex2"<<endl;
    //todo change it to mc_iter 
    //todo python run.py  --opt -2 --size 4 --pivoting 1  --check_equal 1 --verbose  true
    // int mc_iter = 1;//100000;//
    for(int i= 0; i<mc_iter; i++){
      all_equal = run_LU_ex2(kern_size,verbose,pivoting,(bool)check_equal);
      if ((all_equal != 1)&& check_equal){
        cout<<"warning::: not equal"<<endl;
        exit(-1);
      }
    }
  }else{
    
    run_vec_add();
  }
   
  return 0;
}
