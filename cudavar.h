#include "hparams.h"
namespace CUDA
{
    template<typename T>
    class Var
    {
        protected:
        T *t;
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
        void Init()
        {
            cudaMalloc((void **)&t, sizeof(T));
            autofreeon();
        }
        void assign(T x)
        {
            cudaMemcpy(t, &x, sizeof(T), cudaMemcpyHostToDevice);
        }
        T get()
        {
            T x;
            cudaMemcpy(&x,t, sizeof(T), cudaMemcpyDeviceToHost);
            return x;
        }
        public:
        Var()
        {
            Init();
        }
        Var(T x)
        {
            Init();
            assign(x);
        }
        Var(const Var &v)
        {
            t = v.t;
            autofree = v.autofree;
        }
        void operator=(T x)
        {
            assign(x);
        }
        operator T()
        {
            return get();
        }
        void free()
        {
            cudaFree(t);
        }
        ~Var()
        {
            if(autofree)
                free();
            else autofreeon();
        }
        Var operator+(Var v)
        {
            Var x;
            x.autofreeoff();
            bin_op<T><<<1,1>>>(this->t,v.t,x.t,c_add<T>);
            return x;
        }
    };
};