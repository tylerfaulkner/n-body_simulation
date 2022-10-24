#include <stdio.h>
#include <stdlib.h>
#include <curand.h> //cuda random library

__global__ void initialzeBodies(void *d_X) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float4 *globalX = (float *)d_X;

    /*
    TODO
    */
    //float xPos = 
}