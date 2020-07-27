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
