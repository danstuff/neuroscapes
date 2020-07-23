#include "util.h"

float NtoF(neum v){
    //convert neum into a float
    return ((float) v) / ((float) NEUM_LIM);
}

neum FtoN(float v){
    //convert float into a space-saving neum
    if(v > 1){ v = 1; }
    if(v < 0){ v = 0; }
    return (neum) (v * NEUM_LIM);
}

float randf(float min, float max){
    return (rand() % ((max-min)*RAND_MUL))/RAND_MUL + min;
}

float sig(float z){
    //sigmoid
    return 1 / (1 + exp(z));
}

float sigp(float z){
    //derivative of sigmoid
    return sig(z)*(1-sig(z));
}

float cost(float a, float y){
    //perform cross-entropy cost function
    if(a == 1 || a == 0){
        return 0;
    }

    return -y*log(a) - (1-y)*log(1-a); 
}

void copy(float* a, float* b, uint16 size){
    for(uint16 i = 0; i < size; i++){
        b[i] = a[i];
    }
}

void copy2d(float* a, float* b, uint16 d, uint16 b){
    for(uint16 i = 0; i < d; i++){
        for(uint16 j = 0; j < b; j++){
            b[i][j] = a[i][j];
        }
    }
}

void transpose(float* arr, float* ans, uint16 d, uint16 b){
    for(uint16 i = 0; i < d; i++){
        for(uint16 j = 0; j < b; j++){
            //swap i,j and j,i in the answer
            if(d == 1 && b == 1){
                ans = arr;
            } else if(b == 1){
                ans[j] = arr[i];
            } else {
                ans[j][i] = arr[i][j];
            }
        }
    }
}

void dot(float* a, float* b, float* ans, uint16 da, uint16 ba, uint16 db, uint16 bb){
    assert(ba == db);

    for(uint16 i = 0; i < da; i++){
        for(uint16 j = 0; j < bb; j++){
            ans[i][j] = 0;

            for(uint16 k = 0; k < ba; k++){
                if(ba == 1 && bb == 1){
                    ans[i][j] += a[i]*b[k];
                } else if(ba == 1){
                    ans[i][j] += a[i]*b[k][j];
                } else if(bb == 1){
                    ans[i][j] += a[i][k]*b[k];
                } else {
                    ans[i][j] += a[i][k]*b[k][j];
                }
            }
        }
    }
}
