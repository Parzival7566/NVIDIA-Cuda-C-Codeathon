#include<stdio.h>
#include <stdlib.h>
#define intswap(A,B) {int temp=A;A=B;B=temp;}

const int N = 1000;
const int threadsPerBlock = 1024;
const int blocksPerGrid = 1024;

__global__ void sort(int *c,int *count)
{
    int l;
    if(*count%2==0)
          l=*count/2;
    else
         l=(*count/2)+1;
    for(int i=0;i<l;i++)
    {
    if(c[threadIdx.x]>c[threadIdx.x+1]) intswap(c[threadIdx.x], c[threadIdx.x+1]);
    __syncthreads();
    }
}



int main()
{
  int a[N],b[N],i;
  for(i=0;i<N;i++)
  a[i]=rand()%200;
  
  printf("ORIGINAL ARRAY : \n");
  for(int i=0;i<N;i++) printf("%d ",a[i]);
  
  int *c,*count;
  
  cudaMalloc((void**)&c,sizeof(int)*N);
  cudaMalloc((void**)&count,sizeof(int));
  
  cudaMemcpy(c,&a,sizeof(int)*N,cudaMemcpyHostToDevice);
  cudaMemcpy(count,&N,sizeof(int),cudaMemcpyHostToDevice);
  
  sort<<< blocksPerGrid,threadsPerBlock >>>(c,count);
  
  cudaMemcpy(&b,c,sizeof(int)*N,cudaMemcpyDeviceToHost);
  
  printf("\nSORTED ARRAY : \n");
  for(int i=0;i<N;i++) printf("%d ",b[i]);
  printf("\n");
}