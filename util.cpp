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

neum sig(float z){
    //sigmoid
    return NEUM_LIM / (1 + exp(z));
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

void copy(neum* a, neum* b, uint16 size){
    for(uint16 i = 0; i < size; i++){
        b[i] = a[i];
    }
}

void transpose(neum* arr, const uint16 size){
    for(uint16 i = 0; i < size; i++){
        for(uint16 j = 0; j < size; j++){
            //swap i,j and j,i
            neum tmp = arr[j][i];
            arr[j][i] = arr[i][j];
            arr[i][j] = tmp;
        }
    }
}
