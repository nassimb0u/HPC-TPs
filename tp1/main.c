#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <pthread.h>
#define SEED 35791246

#define NUM_THREADS 3

long count = 0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

typedef struct parms
{
    int start;
    int end;
} parms;


void *estimation_job(void *arg) {
    parms *prm = (parms*)arg;
    //parms prm = *(parms*)arg;
    int start = prm->start;
    int end = prm->end;
    //int start = prm.start;
    //int end = prm.end;
    double x, y, z;
    long local_count = 0;
    for(int i=start; i<end; i++) {
        x = (double)rand()/RAND_MAX;
        y = (double)rand()/RAND_MAX;
        z = x*x + y*y;
        if(z<=1) local_count++;
    }
    pthread_mutex_lock(&mutex);
    count += local_count;
    pthread_mutex_unlock(&mutex);
}

void main(int argc, char *argv) {
    pthread_t threads[NUM_THREADS];
    int rc;
    parms *prm;
    //parms prms[NUM_THREADS];

    float timeInS;
    clock_t t1, t2;

    int nitr;
    double pi;

    printf("Enter the number of iterations used to estimate pi: ");
    scanf("%d", &nitr);

    /* Begin */
    t1 = clock();
    /* initialize random numbers */
    srand(SEED);

    for(int i=0; i<NUM_THREADS; i++) {
        prm = malloc(sizeof(parms));
        prm->start = i*(int)(nitr/NUM_THREADS);
        prm->end = prm->start + (int)(nitr/NUM_THREADS);
        if(i==NUM_THREADS-1) prm->end = nitr;
        //prms[i].start = i*(int)(nitr/NUM_THREADS);
        //prms[i].end = prms[i].start + (int)(nitr/NUM_THREADS);
        //if(i==NUM_THREADS-1) prms[i].end = nitr;
        rc = pthread_create(&threads[i], NULL, estimation_job, (void *)prm);
    }
    for(int i=0; i<NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    pi = (double)count/nitr*4;
    /* End */
    t2 = clock();
    timeInS = (float)(t2-t1)/CLOCKS_PER_SEC;

    printf("time = %f\n", timeInS);
    printf("# if trials= %d , estimate of pi is %g \n", nitr, pi);
}
