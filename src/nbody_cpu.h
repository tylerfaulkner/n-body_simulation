#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EPS2 0.01 //EPSILON SQUARED dampening factor in textbook, used to prevent values to approach infinity

void calculate_forces(float4 *h_X, float4 *h_A, int bodyCount)
{
    float4 myPosition;
    int i;
    for(int currentBody=0; currentBody<bodyCount; currentBody++){
        float3 acc = {0.0f, 0.0f, 0.0f};
        myPosition = h_X[currentBody];
        for (i = 0; i < bodyCount; i++) {
            acc = bodyBodyInteraction(myPosition, h_X[i], acc);
        }
        float4 acc4 = {acc.x, acc.y, acc.z, 0.0f};
        h_A[currentBody] = acc4;
    }
}

/*
Updates Velocity based on Computed Acceleration
Leapfrog Integration
*/
void calculate_velocity(float4 *h_A, float4 *h_V, int bodyCount, float time)
{
    for(int currentBody=0; currentBody<bodyCount; currentBody++){ 
        float3 vel = {0.0f, 0.0f, 0.0f};
        vel.x = h_V[currentBody].x + h_A[currentBody].x * time;
        vel.y = h_V[currentBody].y + h_A[currentBody].y * time;
        vel.z = h_V[currentBody].z + h_A[currentBody].z * time;
        float4 vel4 = {vel.x, vel.y, vel.z, 0.0f};
        h_V[currentBody] = vel4;
    }
}

/*
Calculate Postion based on Velocity
Leapfrog Integration
*/
void calculate_position(float4 *h_X, float4 *h_V, int bodyCount, float time)
{
    for(int currentBody=0; currentBody<bodyCount; currentBody++){
        float4 currentPos = h_X[currentBody];
        float4 pos = {0.0f, 0.0f, 0.0f, currentPos.w};
        pos.x = currentPos.x + h_V[currentBody].x * time;
        pos.y = currentPos.y + h_V[currentBody].y * time;
        pos.z = currentPos.z + h_V[currentBody].z * time;
        h_X[currentBody] = pos;
    }
}
