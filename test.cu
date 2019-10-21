#include<iostream>
#include "funcs.h"
#include "cudavar.h"
#include "cudavector.h"

void random_ints(int* a, int n)
{
   int i;
   for (i = 0; i < n; ++i)
    a[i] = i;//rand();
}

#define N 1024

int main()
{
    int x=10,y=20;
    CUDA::Var<int> a,b;
    a=x;
    b=y;
    CUDA::Var<int> c;
    c=a+b;
    printf("%d",(int)c);
}