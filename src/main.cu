#include <stdio.h>
#include <stdlib.h>
#include<time.h>  
#include "nbody_kernel.cuh"
#include "nbody_cpu.h"
#include "nbody_init.cuh"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))
#define TIME_STEP 0.25
#define RESULTS_FOLDER "results2"

//handle error macro 
static void HandleError(cudaError_t err, const char *file,  int line ) { 
        if (err != cudaSuccess) { 
            printf("%s in %s at line %d\n", cudaGetErrorString(err),  file, line ); 
        } 
}

// elapsed time in milliseconds
float cpu_time(timespec* start, timespec* end){
        return ((1e9*end->tv_sec + end->tv_nsec) - (1e9*start->tv_sec + start->tv_nsec))/1e6;
}


void outputToFile(double4 *h_X, int bodyCount, float time){
    mkdir(RESULTS_FOLDER, 0777);
    FILE *fp;
    char filename[30];
    sprintf(filename, "%s/%#.2f.csv", RESULTS_FOLDER, time);
    fp = fopen(filename, "w");
    if (fp==NULL){
        printf("Write Error occured");
        return;
    }
    for(int i =0; i<bodyCount; i++){
        double4 bodyPositon = h_X[i];
        fprintf(fp, "%f, %f, %f, %f\n", bodyPositon.x, bodyPositon.y, bodyPositon.z, bodyPositon.w);
    }
}

int main(int argc, char* argv[]) {
    if(argc > 3){
        printf("There are too many arguments.\n");
        return 0;
    }

    if(argc<3){
        printf("Please provide the amount of bodies and simulation steps as a command line argument.\n");
        return 0;
    }
    int n = atoi(argv[1]);
    int k = atoi(argv[2]);
    
    double4 *d_A, *d_X, *d_V, *h_X, *h_A, *h_V;

    size_t size = n*sizeof(double4);

    HANDLE_ERROR(cudaMalloc((void **)&d_X, size));
    HANDLE_ERROR(cudaMalloc((void **)&d_A, size));
    HANDLE_ERROR(cudaMalloc((void **)&d_V, size));

    HANDLE_ERROR(cudaMemset(d_A, 0, size));
    HANDLE_ERROR(cudaMemset(d_V, 0, size));

    h_X = (double4 *)malloc(size);
    h_A = (double4 *)malloc(size);
    h_V = (double4 *)malloc(size);
    memset(h_A, 0, size);
    memset(h_V, 0, size);

    timespec ts, te;

    printf("Randomizing Body Start Positions...\n");
    srand(time(0));
    initializeBodies(h_X, n);

    printf("Verifying Randomization:\n\tx:%lf, y:%lf, z:%lf, w:%lf\n", h_X[0].x,h_X[0].y,h_X[0].z,h_X[0].w);

    // Start benchmark
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    for(int step=0; step<k; step++){
        if(step % 10 == 0){
            printf("Executing Step %d out of %d\n", step, k);
        }
        calculate_forces(h_X, h_A, n);
        //calculate new positions (0.25 is the change in time. We are doing 1/4 a second for each step.)
        calculate_velocity(h_A, h_V, n, TIME_STEP);
        calculate_position(h_X, h_V, n, TIME_STEP);
        //output positions to csv file
        outputToFile(h_X, n, step*TIME_STEP);
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &te);
    float simTime = cpu_time(&ts, &te);
    printf("\nCPU implementation elapsed time: %f ms\n", simTime);
    printf("Single Step average execution time: %f ms\n", simTime/k);

    free(h_X);
    free(h_A);
    free(h_V);
    cudaFree(d_A);
    cudaFree(d_V);
    cudaFree(d_X);
}
