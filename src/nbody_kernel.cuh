#include <stdio.h>
#include <stdlib.h>

#define EPS2 0.01 //dampening factor in textbook

__device__ __host__ float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai)
{
    float3 r;
    // r_ij [3 FLOPS]
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;
    // distSqr = dot(r_ij, r_ij) + EPS^2 [6 FLOPS]
    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;
    // invDistCube =1/distSqr^(3/2) [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    float distSixth = distSqr * distSqr * distSqr;
    float invDistCube = 1.0f/sqrtf(distSixth);
    // s = m_j * invDistCube [1 FLOP]
    float s = bj.w * invDistCube;
    // a_i = a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;
    return ai;
}

__device__ float3 tile_calculation(float4 myPosition, float3 accel, int bodiesLeft)
{
    int i;
    extern __shared__ float4 shPosition[];
    for (i = 0; i < blockDim.x && i < bodiesLeft; i++) {
        accel = bodyBodyInteraction(myPosition, shPosition[i], accel); 
    }
    return accel;
}

__global__ void gpu_calculate_forces(void *d_X, void *d_A, int n)
{
    extern __shared__ float4 shPosition[];
    float4 *globalX = (float4 *)d_X;
    float4 *globalA = (float4 *)d_A;
    float4 myPosition;
    int i, tile;
    float3 acc = {0.0f, 0.0f, 0.0f};
    
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    //Tiling
    for (i = 0, tile = 0; i < n; i += blockDim.x, tile++) {
        int idx = tile * blockDim.x + threadIdx.x;
        if(idx < n){
            shPosition[threadIdx.x] = globalX[idx];
        }
        __syncthreads();
        if(gtid < n){
            myPosition = globalX[gtid];
            acc = tile_calculation(myPosition, acc, n-i);
        }
        __syncthreads();
    }
    // Save the result in global memory for the integration step.
    if(gtid < n){
        float4 acc4 = {acc.x, acc.y, acc.z, 0.0f};
        globalA[gtid] = acc4;
    }
}

__global__ void tileless_gpu_calculate_forces(float4 *d_X, float4 *d_A, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n){
        float4 myPosition = d_X[id];
        float3 acc = {0.0f, 0.0f, 0.0f};
        for(int i=0; i<n; i++){
            acc = bodyBodyInteraction(myPosition, d_X[i], acc);
        }
        float4 acc4 = {acc.x, acc.y, acc.z, 0.0f};
        d_A[id] = acc4;
    }
}

/*
Updates Velocity based on Computed Acceleration
Leapfrog Integration
*/
__global__ void gpu_calculate_velocity(float4 *d_A, float4 *d_V, int bodyCount, float time)
{
    int currentBody = blockIdx.x * blockDim.x + threadIdx.x;
    if (currentBody<bodyCount){ 
        float3 vel = {0.0f, 0.0f, 0.0f};
        vel.x = d_V[currentBody].x + d_A[currentBody].x * time;
        vel.y = d_V[currentBody].y + d_A[currentBody].y * time;
        vel.z = d_V[currentBody].z + d_A[currentBody].z * time;
        float4 vel4 = {vel.x, vel.y, vel.z, 0.0f};
        d_V[currentBody] = vel4;
    }
}

/*
Calculate Postion based on Velocity
Leapfrog Integration
*/
__global__ void gpu_calculate_position(float4 *d_X, float4 *d_V, int bodyCount, float time)
{
    int currentBody = blockIdx.x * blockDim.x + threadIdx.x;
    if (currentBody<bodyCount){ 
        float4 currentPos = d_X[currentBody];
        float4 pos = {0.0f, 0.0f, 0.0f, currentPos.w};
        pos.x = currentPos.x + d_V[currentBody].x * time;
        pos.y = currentPos.y + d_V[currentBody].y * time;
        pos.z = currentPos.z + d_V[currentBody].z * time;
        d_X[currentBody] = pos;
    }
}

