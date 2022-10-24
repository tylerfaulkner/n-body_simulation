#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EPS2 0.5 //dampening factor in textbook

float3 bodyBodyInteraction_cpu(float4 bi, float4 bj, float3 ai)
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
    float invDistCube = 1.0f/sqrt(distSixth);
    // s = m_j * invDistCube [1 FLOP]
    float s = bj.w * invDistCube;
    // a_i = a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;
    return ai;
}

void calculate_forces(float4 *h_X, float4 *h_A, int bodyCount)
{
    float4 myPosition;
    int i;
    float3 acc = {0.0f, 0.0f, 0.0f};
    for(int currentBody=0; currentBody<bodyCount; currentBody++){
        myPosition = h_X[currentBody];
        for (i = 0; i < bodyCount; i++) {
            acc = bodyBodyInteraction_cpu(myPosition, h_X[i], acc);
        }
        float4 acc4 = {acc.x, acc.y, acc.z, 0.0f};
        h_A[currentBody] = acc4;
    }
}
