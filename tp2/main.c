#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

#define NUM_THREADS 4

void carre(long *tab, int size)
{
    #pragma omp for
    for(int i=0; i<size; i++) tab[i] *= tab[i];
}

int main()
{
    long *tab;
    int size;
    printf("Entrez la taille du tableau: ");
    scanf("%d", &size);
    tab = malloc(size*sizeof(long));
    srand(time(NULL));
    long sum = 0;
    double start = omp_get_wtime();
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        #pragma omp for
        for(int i=0; i<size; i++) tab[i] = rand();
        carre(tab, size);
        #pragma omp for reduction(+:sum)
        for(int i=0; i<size; i++) sum += tab[i];
    }
    double end = omp_get_wtime();
    printf("sum = %ld time =  %fs\n", sum, end-start);
    free(tab);
    return 0;
}
