#pragma once

#ifndef NEUNET_H
#define NEUNET_H

#include "neuron.h"

//must be at least 3
const uint NEUNET_NUM_LAYERS = 6;

class NeuNet{
    private:
        Neuron layers[NEUNET_MAX_LAYER][NEU_MAX_CON];

        uint num_inputs;
        uint num_outputs;

    public:
        NeuNet(uint num_in, uint num_out);

        void randomizeWeights();

        uint getNumInputs();
        uint getNumOutputs();

        void run(uint* inputs);

        uint getOutput(uint i);

        void backPropagate(uint* expected);
};

#endif
