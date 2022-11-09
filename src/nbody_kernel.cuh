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

__global__ void tileless_gpu_calculate_forces(double4 *d_X, double4 *d_A)
{
    double4 myPosition;
    double3 acc = {0.0f, 0.0f, 0.0f};
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    myPosition = d_X[id];
    acc = bodyBodyInteraction(myPosition, d_X[id], acc);
    double4 acc4 = {acc.x, acc.y, acc.z, 0.0f};
    d_A[id] = acc4;
}

/*
Updates Velocity based on Computed Acceleration
Leapfrog Integration
*/
__global__ void gpu_calculate_velocity(double4 *d_A, double4 *d_V, int bodyCount, double time)
{
    int currentBody = blockIdx.x * blockDim.x + threadIdx.x;
    if (currentBody<bodyCount){ 
        double3 vel = {0.0f, 0.0f, 0.0f};
        vel.x = d_V[currentBody].x + d_A[currentBody].x * time;
        vel.y = d_V[currentBody].y + d_A[currentBody].y * time;
        vel.z = d_V[currentBody].z + d_A[currentBody].z * time;
        double4 vel4 = {vel.x, vel.y, vel.z, 0.0f};
        d_V[currentBody] = vel4;
    }
}

/*
Calculate Postion based on Velocity
Leapfrog Integration
*/
__global__ void gpu_calculate_position(double4 *d_X, double4 *d_V, int bodyCount, double time)
{
    int currentBody = blockIdx.x * blockDim.x + threadIdx.x;
    if (currentBody<bodyCount){ 
        double4 currentPos = d_X[currentBody];
        double4 pos = {0.0f, 0.0f, 0.0f, currentPos.w};
        pos.x = currentPos.x + d_V[currentBody].x * time;
        pos.y = currentPos.y + d_V[currentBody].y * time;
        pos.z = currentPos.z + d_V[currentBody].z * time;
        d_X[currentBody] = pos;
    }
}

