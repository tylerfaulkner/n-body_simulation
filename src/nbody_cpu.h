#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EPS2 0.01 //EPSILON SQUARED dampening factor in textbook, used to prevent values to approach infinity

double3 bodyBodyInteraction_cpu(double4 bi, double4 bj, double3 ai)
{
    double3 r; //vector between bodies
    // r_ij [3 FLOPS]
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;
    // distSqr = dot(r_ij, r_ij) + EPS^2 [6 FLOPS]
    double distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;
    // invDistCube =1/distSqr^(3/2) [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    double distSixth = distSqr * distSqr * distSqr;
    double invDistCube = 1.0f/sqrt(distSixth);
    // s = m_j * invDistCube [1 FLOP]
    double s = bj.w * invDistCube;
    // a_i = a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;
    return ai;
}

void calculate_forces(double4 *h_X, double4 *h_A, int bodyCount)
{
    double4 myPosition;
    int i;
    for(int currentBody=0; currentBody<bodyCount; currentBody++){
        double3 acc = {0.0f, 0.0f, 0.0f};
        myPosition = h_X[currentBody];
        for (i = 0; i < bodyCount; i++) {
            acc = bodyBodyInteraction_cpu(myPosition, h_X[i], acc);
        }
        double4 acc4 = {acc.x, acc.y, acc.z, 0.0f};
        h_A[currentBody] = acc4;
    }
}

/*
Updates Velocity based on Computed Acceleration
Leapfrog Integration
*/
void calculate_velocity(double4 *h_A, double4 *h_V, int bodyCount, double time)
{
    for(int currentBody=0; currentBody<bodyCount; currentBody++){ 
        double3 vel = {0.0f, 0.0f, 0.0f};
        vel.x = h_V[currentBody].x + h_A[currentBody].x * time;
        vel.y = h_V[currentBody].y + h_A[currentBody].y * time;
        vel.z = h_V[currentBody].z + h_A[currentBody].z * time;
        double4 vel4 = {vel.x, vel.y, vel.z, 0.0f};
        h_V[currentBody] = vel4;
    }
}

/*
Calculate Postion based on Velocity
Leapfrog Integration
*/
void calculate_position(double4 *h_X, double4 *h_V, int bodyCount, double time)
{
    for(int currentBody=0; currentBody<bodyCount; currentBody++){
        double4 currentPos = h_X[currentBody];
        double4 pos = {0.0f, 0.0f, 0.0f, currentPos.w};
        pos.x = currentPos.x + h_V[currentBody].x * time;
        pos.y = currentPos.y + h_V[currentBody].y * time;
        pos.z = currentPos.z + h_V[currentBody].z * time;
        h_X[currentBody] = pos;
    }
}
