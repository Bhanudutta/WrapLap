#include "hparams.h"
namespace CUDA
{
    template<typename T>
    class Vector
    {
        protected:
        T *t;
        int size;
        bool autofree;
        private:
        void autofreeon()
        {
            autofree = true;
        }
        void autofreeoff()
        {
            autofree = false;
        }
        void Init(int s)
        {
            size = s;
            cudaMalloc((void **)&t, sizeof(T)*size);
            autofreeon();
        }
        void assign(T *x)
        {
            cudaMemcpy(t, x, sizeof(T)*size, cudaMemcpyHostToDevice);
        }
        T* get()
        {
            T *x;
            x = new T[size];
            cudaMemcpy(x,t, sizeof(T)*size, cudaMemcpyDeviceToHost);
            return x;
        }
        public:
        Vector(int s)
        {
            Init(s);
        }
        Vector(T *x,int s)
        {
            Init(s);
            assign(x);
        }
        Vector(const Vector &v)
        {
            t = v.t;
            size = v.size;
            autofree = v.autofree;
        }
        void operator=(T *x)
        {
            assign(x);
        }
        operator T*()
        {
            return get();
        }
        void free()
        {
            cudaFree(t);
        }
        ~Vector()
        {
            if(autofree)
                free();
            else autofreeon();
        }
        Vector operator+(Vector v)
        {
            Vector x(size);
            x.autofreeoff();
            int LN = size%THREADS_PER_BLOCK;
            int K = size/THREADS_PER_BLOCK;
            v_bin_op<T><<<K,THREADS_PER_BLOCK>>>(this->t,v.t,x.t,c_add<T>);
            v_bin_op<T><<<1,LN>>>(this->t,v.t,x.t,c_add<T>);
            return x;
        }
    };
};