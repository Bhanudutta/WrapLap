#include "hparams.h"
namespace CUDA
{
 
    template<typename T>
    using bi_func = T (*) (T, T);
 

    //basic binary functions device codes
    template<typename T>
    __device__ T add(T a,T b)
    {
        return a+b;
    }

    template<typename T>
    __device__ T sub(T a,T b)
    {
        return a-b;
    }
    template<typename T>
    __device__ T mul(T a,T b)
    {
        return a*b;
    }
    template<typename T>
    __device__ T div(T a,T b)
    {
        return a/b;
    }
 
    template<typename T>
    __device__ bi_func<T> c_add = add<T>;



    //binary operation kernels
    template<typename T>
    __global__ void bin_op(T *a,T *b,T *c,bi_func<T> fun)
    {
        printf("%d",10);
        *c = (*fun)(*a,*b);
        printf("%d",10);
    }

    template<typename T>
    __global__ void v_bin_op(T *a,T *b,T *c,bi_func<T> fun)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        c[index] = (*fun)(a[index],b[index]);
    }
};