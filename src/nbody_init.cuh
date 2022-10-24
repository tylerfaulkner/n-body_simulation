#include <stdio.h>
#include <stdlib.h>

void initializeBodies(float4 *h_X, int n) {
    for(int i=0; i<n; i++){
        float xPos = (float)rand();
        float yPos = (float)rand();
        float zPos = (float)rand();
        float mass = (float)rand();
        h_X[i] = {xPos, yPos, zPos, mass};
    }
}