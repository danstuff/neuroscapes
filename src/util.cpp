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
