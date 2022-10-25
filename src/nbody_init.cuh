#include <stdio.h>
#include <stdlib.h>

#define MAX_DIM 50
#define MOD_DIM MAX_DIM*2

void initializeBodies(double4 *h_X, int n) {
    for(int i=0; i<n; i++){
        double xPos = (double)((rand() % MOD_DIM)-MAX_DIM);
        double yPos = (double)((rand() % MOD_DIM)-MAX_DIM);
        double zPos = (double)((rand() % MOD_DIM)-MAX_DIM);
        double mass = (double)((rand() % MOD_DIM)-MAX_DIM);
        h_X[i] = {xPos, yPos, zPos, mass};
    }
}