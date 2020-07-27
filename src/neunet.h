#pragma once

#ifndef NEUNET_H
#define NEUNET_H

#include "util.h"

const uint16 NEUNET_INPUTS = 3;
const uint16 NEUNET_OUTPUTS = 3;

const uint16 NEUNET_DEPTH = 4; //must be at least 3 
const uint16 NEUNET_BREADTH = 3;

class NeuNet{
    private:
        neum biases[NEUNET_DEPTH-1][NEUNET_BREADTH];
        neum weights[NEUNET_DEPTH-2][NEUNET_BREADTH][NEUNET_BREADTH];

        float getBias(uint16 layer, uint16 i);
        float getWeight(uint16 layer, uint16 i, uint16 j);

        void setBias(uint16 layer, uint16 i, float v);
        void setWeight(uint16 layer, uint16 i, uint16 j, float v);
        
        uint16 getLyrBreadth(uint16 l);

        float getZ(uint16 l, uint16 i, float* p);

    public:
        NeuNet();

        void feedfwd(float* a);
        void backprop(float* a, float* y);
};

#endif
