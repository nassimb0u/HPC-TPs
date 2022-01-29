#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define SEED 35791246

void main(int argc, char *argv) {
    float timeInS;
    clock_t t1, t2;

    int nitr;
    double x, y, z;
    long count = 0;
    double pi;

    printf("Enter the number of iterations used to estimate pi: ");
    scanf("%d", &nitr);

    /* Begin */
    t1 = clock();
    /* initialize random numbers */
    srand(SEED);

    for(int i=0; i<nitr; i++) {
        x = (double)rand()/RAND_MAX;
        y = (double)rand()/RAND_MAX;
        z = x*x + y*y;
        if(z<=1) count++;
    }

    pi = (double)count/nitr*4;
    /* End */
    t2 = clock();
    timeInS = (float)(t2-t1)/CLOCKS_PER_SEC;

    printf("time = %f\n", timeInS);
    printf("# if trials= %d , estimate of pi is %g \n", nitr, pi);
}
