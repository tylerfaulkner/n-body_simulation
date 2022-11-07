#include <stdio.h>
#include <stdlib.h>

#define EPS2 0.01 //dampening factor in textbook
#define TILE_WIDTH 32


__device__ double3 bodyBodyInteraction(double4 bi, double4 bj, double3 ai)
{
    double3 r;
    // r_ij [3 FLOPS]
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;
    // distSqr = dot(r_ij, r_ij) + EPS^2 [6 FLOPS]
    double distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;
    // invDistCube =1/distSqr^(3/2) [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    double distSixth = distSqr * distSqr * distSqr;
    double invDistCube = 1.0f/sqrtf(distSixth);
    // s = m_j * invDistCube [1 FLOP]
    double s = bj.w * invDistCube;
    // a_i = a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;
    return ai;
}

__device__ double3 tile_calculation(double4 myPosition, double3 accel)
{
    int i;
    extern __shared__ double4 shPosition[];
    for (i = 0; i < blockDim.x; i++) {
        accel = bodyBodyInteraction(myPosition, shPosition[i], accel); 
    }
    return accel;
}

__global__ void gpu_calculate_forces(void *devX, void *devA, int n)
{
    extern __shared__ double4 shPosition[];
    double4 *globalX = (double4 *)devX;
    double4 *globalA = (double4 *)devA;
    double4 myPosition;
    int i, tile;
    double3 acc = {0.0f, 0.0f, 0.0f};
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    myPosition = globalX[gtid];
    //Tiling
    for (i = 0, tile = 0; i < n; i += TILE_WIDTH, tile++) {
        int idx = tile * blockDim.x + threadIdx.x;
        shPosition[threadIdx.x] = globalX[idx];
        __syncthreads();
        acc = tile_calculation(myPosition, acc);
        __syncthreads();
    }
    // Save the result in global memory for the integration step.
    double4 acc4 = {acc.x, acc.y, acc.z, 0.0f};
    globalA[gtid] = acc4;
}

