#include <stdio.h>
#include <stdlib.h>

#define N 16
#define M 16

__global__ void prod(int *a, int *b, int *c) {
    const int t = 16;
    __shared__ int tmp[t];
    int i = blockIdx.y;
    int j = threadIdx.x;
    tmp[j] = a[i * M + j] * b[j];
    __syncthreads();
    if (threadIdx.x == 0) {
        int sum = 0;
        for (int k = 0; k < 16; k++) sum += tmp[k];
        atomicAdd(c + i, sum);
    }
}

int main() {
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    dim3 grid(1, 16);
    dim3 block(16, 1);
    // variables pour le calcule du temps
    cudaEvent_t start, stop;
    float elapsed_time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // allocation
    a = (int *)malloc(N * M * sizeof(int));
    b = (int *)malloc(M * sizeof(int));
    c = (int *)malloc(N * sizeof(int));
    // init
    for (int i = 0; i < N * M; i++) a[i] = (i / M) == (i % M);
    for (int i = 0; i < M; i++) b[i] = i;
    // allocation mem GPU
    cudaMalloc((void **)&d_a, N * M * sizeof(int));
    cudaMalloc((void **)&d_b, M * sizeof(int));
    cudaMalloc((void **)&d_c, N * sizeof(int));
    // cpy data
    cudaMemcpy(d_a, a, N * M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, M * sizeof(int), cudaMemcpyHostToDevice);
    // call kernel
    cudaEventRecord(start, 0);
    prod<<<grid, block>>>(d_a, d_b, d_c);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Elapsed time : %f ms\n", elapsed_time);
    // cpy results
    cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);
    // show results
    printf("c = ");
    for (int i = 0; i < N; i++) printf("%d ", c[i]);
    printf("\n");
    // free mem gpu
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    // free mem
    free(a);
    free(b);
    free(c);
    return 0;
}
