#pragma once
#include <iostream>
using std::cout; 
using std::endl;
void print_mat(float* h_A, int M, int N);

void print_mat2(float* h_A, int M, int N);

void mat_mul(float* LL, float* UU, float* LLUU, int M, int N);

int check_equal(float* LLUU,float* in_mat, int M, int N);
int check_equal2(float* LLUU,float* in_mat, int M, int N);