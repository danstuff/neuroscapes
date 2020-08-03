#pragma once

#ifndef NEUNET_H
#define NEUNET_H

#include "include/util.h"
#include "include/matrix.h"

const uint16 NEUNET_INPUTS = 32;
const uint16 NEUNET_OUTPUTS = 2;

const uint16 NEUNET_DEPTH = 26;
const uint16 NEUNET_BREADTH = 32;

class NeuNet{
    private:
        float biases[NEUNET_DEPTH][NEUNET_BREADTH];
        float weights[NEUNET_DEPTH][NEUNET_BREADTH][NEUNET_BREADTH];
        
        uint16 getLyrBreadth(uint16 l);

    public:
        NeuNet();

        void print();

        void feedfwd(Matrix& a, Matrix* a_collect = NULL, Matrix* z_collect = NULL);
        void backprop(Matrix* trial_as, Matrix* trial_ys, 
                      uint16 num_trials, uint16 total_size, 
                      float learn_rate, float reg);
};

#endif
