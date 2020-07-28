#pragma once

#ifndef NEUNET_H
#define NEUNET_H

#include "include/util.h"
#include "include/matrix.h"

const uint16 NEUNET_INPUTS = 3;
const uint16 NEUNET_OUTPUTS = 3;

const uint16 NEUNET_DEPTH = 4;
const uint16 NEUNET_BREADTH = 3;

class NeuNet{
    private:
        neum biases[NEUNET_DEPTH][NEUNET_BREADTH];
        neum weights[NEUNET_DEPTH][NEUNET_BREADTH][NEUNET_BREADTH];
        
        uint16 getLyrBreadth(uint16 l);

    public:
        NeuNet();

        void feedfwd(Matrix& a, Matrix* a_collect = NULL, Matrix* z_collect = NULL);
        void backprop(Matrix& a, Matrix& y);
};

#endif
