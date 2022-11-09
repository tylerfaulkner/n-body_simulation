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
#define BLOCK_SIZE 1024

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


void outputToFile(float4 *h_X, int bodyCount, float time){
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
        float4 bodyPositon = h_X[i];
        fprintf(fp, "%f, %f, %f, %f\n", bodyPositon.x, bodyPositon.y, bodyPositon.z, bodyPositon.w);
    }
}

void cpuKernel(float4 *h_X, float4 *h_A, float4 *h_V, int n, int k, bool outputResults){
    for(int step=0; step<k; step++){
        calculate_forces(h_X, h_A, n);
        //calculate new positions (0.25 is the change in time. We are doing 1/4 a second for each step.)
        calculate_velocity(h_A, h_V, n, TIME_STEP);
        calculate_position(h_X, h_V, n, TIME_STEP);
        //output positions to csv file
        if (outputResults) 
            outputToFile(h_X, n, step*TIME_STEP);
    }
}

void copyfloatArray(float4 *h_destination, float4 *h_source, int elements){
    for (int i = 0; i < elements; i++){
        h_destination[i] = {h_source[i].x, h_source[i].y, h_source[i].z, h_source[i].w};
    }
}

void verifyOutput(float4 *h_X, float4 *h_XfromDevice, int n){
    for(int i = 0; i<n; i++){
        if (h_X[i].x != h_XfromDevice[i].x){
            printf("Device results do not equal CPU results for body %i\n", i);
            printf("%lf != %lf\n", h_X[i].x, h_XfromDevice[i].x);
        }
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
    
    float4 *d_A, *d_X, *d_V, *h_X, *h_A, *h_V, *h_OriginalCopy;

    size_t size = n*sizeof(float4);

    HANDLE_ERROR(cudaMalloc((void **)&d_X, size));
    HANDLE_ERROR(cudaMalloc((void **)&d_A, size));
    HANDLE_ERROR(cudaMalloc((void **)&d_V, size));

    HANDLE_ERROR(cudaMemset(d_A, 0, size));
    HANDLE_ERROR(cudaMemset(d_V, 0, size));

    h_X = (float4 *)malloc(size);
    h_A = (float4 *)malloc(size);
    h_V = (float4 *)malloc(size);
    h_OriginalCopy = (float4 *)malloc(size);

    memset(h_A, 0, size);
    memset(h_V, 0, size);

    timespec ts, te;

    printf("Randomizing Body Start Positions...\n");
    srand(time(0));
    initializeBodies(h_X, n);
    copyfloatArray(h_OriginalCopy, h_X, n);

    //printf("Verifying Randomization:\n\tx:%lf, y:%lf, z:%lf, w:%lf\n", h_X[0].x,h_X[0].y,h_X[0].z,h_X[0].w);

    // Start benchmark
    printf("Starting CPU Kernel...");
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    cpuKernel(h_X, h_A, h_V, n, k, OUTPUT_TO_FILE);
    clock_gettime(CLOCK_MONOTONIC_RAW, &te);

    float simTime = cpu_time(&ts, &te);
    printf("\nCPU implementation elapsed time: %f ms\n", simTime);
    //printf("Single Step average execution time: %f ms\n", simTime/k);


    // Start GPU Implementation
    printf("\nStarting GPU Implementation\n");
    int block_in_grid = (int)ceil( (float)n / BLOCK_SIZE);

    HANDLE_ERROR(cudaMemcpy(d_X, h_OriginalCopy, size, cudaMemcpyHostToDevice));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
    for(int step=0; step<k; step++){
        gpu_calculate_forces<<<block_in_grid, BLOCK_SIZE, BLOCK_SIZE*sizeof(float4)>>>(d_X, d_A, n);
        cudaDeviceSynchronize();
        gpu_calculate_velocity<<<block_in_grid, BLOCK_SIZE>>>(d_A, d_V, n, TIME_STEP);
        cudaDeviceSynchronize();
        gpu_calculate_position<<<block_in_grid, BLOCK_SIZE>>>(d_X, d_V, n, TIME_STEP);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU Implementation Elapsed time: %f ms\n", milliseconds);

    float4 *d_Xnt;
    HANDLE_ERROR(cudaMalloc((void **)&d_Xnt, size));
    HANDLE_ERROR(cudaMemcpy(d_Xnt, h_OriginalCopy, size, cudaMemcpyHostToDevice));

    cudaFree(d_V);
    cudaMemset(d_V, 0, size);

    cudaEvent_t startNT, stopNT;
	cudaEventCreate(&startNT);
	cudaEventCreate(&stopNT);

    cudaEventRecord(startNT);
    for(int step=0; step<k; step++){
        tileless_gpu_calculate_forces<<<block_in_grid, BLOCK_SIZE>>>(d_Xnt, d_A, n);
        cudaDeviceSynchronize();
        gpu_calculate_velocity<<<block_in_grid, BLOCK_SIZE>>>(d_A, d_V, n, TIME_STEP);
        cudaDeviceSynchronize();
        gpu_calculate_position<<<block_in_grid, BLOCK_SIZE>>>(d_Xnt, d_V, n, TIME_STEP);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stopNT);

    cudaEventSynchronize(stopNT);
    float millisecondsNT = 0;
	cudaEventElapsedTime(&millisecondsNT, startNT, stopNT);
	printf("GPU Non-Tiled Implementation Elapsed time: %f ms\n", millisecondsNT);
    
    float4 *h_XfromDevice;
    h_XfromDevice = (float4 *)malloc(size);
    HANDLE_ERROR(cudaMemcpy(h_XfromDevice, d_Xnt, size, cudaMemcpyDeviceToHost));

    verifyOutput(h_X, h_XfromDevice, n);

    free(h_X);
    free(h_A);
    free(h_V);
    cudaFree(d_A);
    cudaFree(d_V);
    cudaFree(d_X);
}
