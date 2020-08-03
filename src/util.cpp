#include "include/util.h"

float randf(float min, float max){
    return (rand() % ((uint32)(max-min)*RAND_MUL))/((float)RAND_MUL) + min;
}

float sigmoid(float z){
    //sigmoid
    return 1 / (1 + exp(-z));
}

float sigmoidp(float z){
    //derivative of sigmoid
    return sigmoid(z)*(1-sigmoid(z));
}

float truncate(float v){
    //cut off the end of a floating point number by int casting it
    return ((float)((int)(v*ROUND_MUL))/ROUND_MUL);
}
