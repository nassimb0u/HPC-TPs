#include <stdio.h>
#include <stdlib.h>

#define N 16
#define M 16

__global__ void prod(int **a, int *b, int *c) {
    const int t = 16;
    __shared__ int tmp[t];
    int i = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    int j = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    tmp[i] = a[i][j] * b[j] + a[i][j + 1] * b[j + 1];
    tmp[i + 1] = a[i + 1][j] * b[j] + a[i + 1][j + 1] * b[j + 1];
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int k = 0; k < 4; k++) atomicAdd(&c[i + k], tmp[i + k]);
    }
}

int main() {
    int **a, *b, *c;
    int **d_a, *d_b, *d_c;
    dim3 grid(4, 4);
    dim3 block(2, 2);
    // variables pour le calcule du temps
    cudaEvent_t start, stop;
    float elapsed_time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // allocation
    a = (int **)malloc(N * sizeof(int *));
    for (int i = 0; i < N; i++) a[i] = (int *)malloc(M * sizeof(int));
    b = (int *)malloc(M * sizeof(int));
    c = (int *)malloc(N * sizeof(int));
    // init
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) a[i][j] = i == j;
    }
    for (int i = 0; i < M; i++) b[i] = i;
    // allocation mem GPU
    cudaMalloc((void **)&d_a, N * sizeof(int *));
    for (int i = 0; i < N; i++) cudaMalloc((void **)&d_a[i], M * sizeof(int));
    cudaMalloc((void **)&d_b, M * sizeof(int));
    cudaMalloc((void **)&d_c, N * sizeof(int));
    // cpy data
    for (int i = 0; i < N; i++)
        cudaMemcpy(d_a[i], a[i], M * sizeof(int), cudaMemcpyHostToDevice);
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
    for (int i = 0; i < N; i++) cudaFree(d_a[i]);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    // free mem
    for (int i = 0; i < N; i++) free(a[i]);
    free(a);
    free(b);
    free(c);
    return 0;
}
