#pragma once

#ifndef NEUNET_H
#define NEUNET_H

#include "neuron.h"

const uint NEUNET_MAX_LAYER = 64;
const uint NEUNET_MAX_NEU = 512;

class NeuNet{
    private:
        Neuron layers[NEUNET_MAX_LAYER][NEUNET_MAX_NEU];

    public:
        NeuNet();

};

#endif
