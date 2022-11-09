#include <stdio.h>
#include <stdlib.h>

#define MAX_DIM 200 //Generates a position between -100 and 100
#define MOD_DIM MAX_DIM*2
#define MAX_MASS 100 //Generates Mass between 0 and 100

/*
Intializes a random position and mass for every body in the system
*/
void initializeBodies(float4 *h_X, int n) {
    for(int i=0; i<n; i++){
        float xPos = (float)((rand() % MOD_DIM)-MAX_DIM);
        float yPos = (float)((rand() % MOD_DIM)-MAX_DIM);
        float zPos = (float)((rand() % MOD_DIM)-MAX_DIM);
        float mass = (float)((rand() % MAX_MASS));
        h_X[i] = {xPos, yPos, zPos, mass};
    }
}