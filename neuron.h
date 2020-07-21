#pragma once

#ifndef NEURON_H
#define NEURON_H

#include "stdio.h"
#include "stdlib.h"
#include "cassert.h"

using namespace std;

typedef unsigned int uint;

const uint NEU_MAX_CON = 32;

struct NeuInput{
    float value;
    float weight;
}

class Neuron{
    private:
        NeuInput inputs[NEU_MAX_CON];
        uint num_inputs;

        float bias;

    public:
        Neuron(uint num_ipts, float bias);

        void clearInputs();
        void randomizeWeights();

        void setInput(uint i, float value);
        void setInputWeight(uint i, float weight);

        float getOutput();
};

#endif
