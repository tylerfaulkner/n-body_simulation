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
#define RESULTS_FOLDER "results"
#define OUTPUT_TO_FILE false

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

void cpuKernel(double4 *h_X, double4 *h_A, double4 *h_V, int n, int k, bool outputResults){
    for(int step=0; step<k; step++){
        if(step % 10 == 0){
            printf("Executing Step %d out of %d\n", step, k);
        }
        calculate_forces(h_X, h_A, n);
        //calculate new positions (0.25 is the change in time. We are doing 1/4 a second for each step.)
        calculate_velocity(h_A, h_V, n, TIME_STEP);
        calculate_position(h_X, h_V, n, TIME_STEP);
        //output positions to csv file
        if (outputResults) 
            outputToFile(h_X, n, step*TIME_STEP);
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
    cpuKernel(h_X, h_A, h_V, n, k, OUTPUT_TO_FILE);
    clock_gettime(CLOCK_MONOTONIC_RAW, &te);

    float simTime = cpu_time(&ts, &te);
    printf("\nCPU implementation elapsed time: %f ms\n", simTime);
    printf("Single Step average execution time: %f ms\n", simTime/k);


    // Start GPU Implementation
    printf("\nStarting GPU Implementation\n");
    int threads_per_block = 32;//1024;
    int block_in_grid = ceil( float(n) / threads_per_block);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
    for(int step=0; step<k; step++){
        if(step % 10 == 0){
            printf("Executing Step %d out of %d\n", step, k);
        }
        gpu_calculate_forces<<<block_in_grid, threads_per_block, 32*32*sizeof(double4)>>>(d_X, d_A, n);
        gpu_calculate_velocity<<<block_in_grid, threads_per_block>>>(d_A, d_V, n, step*TIME_STEP);
        gpu_calculate_position<<<block_in_grid, threads_per_block>>>(d_X, d_V, n, step*TIME_STEP);

        if (OUTPUT_TO_FILE){
            HANDLE_ERROR(cudaMemcpy(h_X, d_X, n, cudaMemcpyDeviceToHost));
            // printf("GPU TESTING %f, %f, %f\n", h_X[0].x, h_X[0].y, h_X[0].z);
            outputToFile(h_X, n, step*TIME_STEP);
        }
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU Implementation Elapsed time: %f ms\n", milliseconds);
    

    free(h_X);
    free(h_A);
    free(h_V);
    cudaFree(d_A);
    cudaFree(d_V);
    cudaFree(d_X);
}
