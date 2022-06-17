#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

const int N = 1000000;
const int threadsPerBlock = 256;
const int blocksPerGrid = 32;

//Here, we run the dot product function on the GPU

__global__ void prefixsum(float* a, float* c) 
{
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    
    float temp = 0;
    while (tid < N){
        temp += a[tid] + a[tid+1];
        tid += blockDim.x * gridDim.x;
    }
    
    // set the cache values
    cache[cacheIndex] = temp;
    
    // synchronizing threads in this block
    __syncthreads();
    
    // for reductions, threadsPerBlock must be a power of 2 since we use the following code
    int i = blockDim.x/2;
    while (i != 0){
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
        }

    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}


int main (void) {
    float *a, c, *partial_c;
    float *dev_a, *dev_partial_c;
    
    //setting CPU timer
    
    clock_t start1, end1;
    double t;    
    start1 = clock();
    
    // allocating memory on the cpu side
    
    a = (float*)malloc(N*sizeof(float));
    partial_c = (float*)malloc(blocksPerGrid*sizeof(float));
    
    // allocating the memory on the gpu
    
    cudaMalloc((void**)&dev_a, N*sizeof(float));
    //cudaMalloc((void**)&dev_b, N*sizeof(float));
    cudaMalloc((void**)&dev_partial_c, blocksPerGrid*sizeof(float));
    
    // fill in the host memory with data
    for(int i=0; i<N; i++) a[i] = i;
    
    //setting GPU timer
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // copy the arrays 'a' and 'b' to the gpu
    
    cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
    
    cudaEventRecord(start);
    
    prefixsum<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_partial_c);
    
    cudaEventRecord(stop);
    
    // copy the array 'c' back from the gpu to the cpu
    cudaMemcpy(partial_c,dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // finishing up on the cpu side
    c = 0;
    for(int i=0; i<blocksPerGrid; i++) {
        c += partial_c[i];
    }
    double prefixSum[N];
    prefixSum[0] = a[0];
    double x;
    for (int i = 0; i < N; i++)
    {
        prefixSum[i] = prefixSum[i - 1] + a[i];
        x+=prefixSum[i];
    }
    end1 = clock();
    t = ((double) (end1 - start1)) / CLOCKS_PER_SEC;

    printf("CPU sequential value calculated in time = %.6f ms\n",t);
    printf("GPU parallelised value calculated in time = %.6f ms\n",milliseconds);
    printf("Speedup obtained = %.6g\n",t/milliseconds);
    
    // freeing memory on the gpu side
    cudaFree(dev_a);
    cudaFree(dev_partial_c);
    
    // freeing memory on the cpu side
    free(a);
    free(partial_c);
}