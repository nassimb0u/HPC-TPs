#include <stdio.h>
#include <stdlib.h>

#define N 4
#define M 4

int main() {
    int **a, *b, *c;
    a = (int **)malloc(N * sizeof(int *));
    for (int i = 0; i < N; i++) a[i] = (int *)malloc(M * sizeof(int));
    b = (int *)malloc(M * sizeof(int));
    c = (int *)malloc(N * sizeof(int));
    // init
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) a[i][j] = i + j;
    }
    for (int i = 0; i < M; i++)
        b[i] = 1;
    // show results
    printf("c = ");
    for (int i = 0; i < N; i++)
        printf("%d ", c[i]);
    printf("\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) printf("%d ", a[i][j]);
        printf("\n");
    }
    // free mem
    for (int i = 0; i < N; i++) free(a[i]);
    free(a);
    free(b);
    free(c);
    return 0;
}
