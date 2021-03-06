#pragma once

#ifndef NEUNET_H
#define NEUNET_H

#include "util.h"
#include "matrix.h"

const uint16 NEUNET_INPUTS = 24;
const uint16 NEUNET_OUTPUTS = 1;

const uint16 NEUNET_DEPTH = 8;
const uint16 NEUNET_BREADTH = 16;

class NeuNet{
    private:
        float biases[NEUNET_DEPTH][NEUNET_BREADTH];
        float weights[NEUNET_DEPTH][NEUNET_BREADTH][NEUNET_BREADTH];
        
        uint16 getLyrBreadth(uint16 l);

        Matrix fillMatWithWeights(uint16 l);

    public:
        NeuNet();

        void print();

        void write(const char* filename);
        void read(const char* filename);

        void feedfwd(Matrix& a, Matrix* a_collect = NULL, Matrix* z_collect = NULL);
        void backprop(Matrix* trial_as, Matrix* trial_ys, 
                      uint16 num_trials, uint16 total_size, 
                      float learn_rate, float reg);
};

#endif
