#pragma once

#ifndef NEUNET_H
#define NEUNET_H

#include "util.h"

const uint16 NEUNET_DEPTH = 6; //must be at least 3
const uint16 NEUNET_BREADTH = 8;

class NeuNet{
    private:
        neum biases[NEUNET_DEPTH-1][NEUNET_BREADTH];
        neum weights[NEUNET_DEPTH-2][NEUNET_BREADTH][NEUNET_BREADTH];

        uint16 num_inputs;
        uint16 num_outputs;

        neum getBias(uint16 layer, uint16 i);
        neum getWeight(uint16 layer, uint16 i, uint16 j);

        void setBias(uint16 layer, uint16 i, neum v);
        void setWeight(uint16 layer, uint16 i, uint16 j, neum v);
        
        float getZ(uint16 l, uint16 i, neum* p);

    public:
        NeuNet(uint16 num_in, uint16 num_out);

        void feedfwd(neum* a);
        void backprop(neum* y);
};

#endif
