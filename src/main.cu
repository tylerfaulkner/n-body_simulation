#include <stdio.h>
#include <stdlib.h>
#include<time.h>  
#include "nbody_kernel.cuh"
#include "nbody_cpu.h"
#include "nbody_init.cuh"

#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ )) 

//handle error macro 
static void HandleError(cudaError_t err, const char *file,  int line ) { 
        if (err != cudaSuccess) { 
            printf("%s in %s at line %d\n", cudaGetErrorString(err),  file, line ); 
        } 
} 

int main(int argc, char* argv[]) {
    if(argc > 3){
        printf("There are too many arguments");
        return 0;
    }

    if(argc<3){
        printf("Please provide the amount of bodies and simulation steps as a command line argument.");
        return 0;
    }
    int n = atoi(argv[1]);
    int k = atoi(argv[2]);

    float4 *d_A, *d_X, *h_X, *h_A;

    size_t size = k*sizeof(float4);

    HANDLE_ERROR(cudaMalloc((void **)&d_X, size));
    HANDLE_ERROR(cudaMalloc((void **)&d_A, size));

    HANDLE_ERROR(cudaMemset(d_A, 0, size));

    h_X = (float4 *)malloc(size);
    h_A = (float4 *)malloc(size);
    memset(h_A, 0, size);

    printf("Randomizing Body Start Positions...\n");
    srand(time(0));
    initializeBodies(h_X, n);

    printf("Verifying Randomization:\n\tx:%lf, y:%lf, z:%lf, w:%lf\n", h_X[0].x,h_X[0].y,h_X[0].z,h_X[0].w);

    for(int step=0; step<k; step++){
        calculate_forces(h_X, h_A, n);
        //calculate new positions
        //output positions to csv file
    }
}
