#pragma once

#ifndef NEURON_H
#define NEURON_H

#include "stdio.h"
#include "stdlib.h"
#include "cassert.h"

using namespace std;

typedef unsigned int uint;

const uint NEU_MAX_CON = 512;

class Neuron{
    private:
        bool inputs[NEU_MAX_CON];
        uint num_inputs;

        uint threshold;

    public:
        Neuron(): num_inputs(0), threshold(0) {};

        void clearInputs();

        void setInput(uint i);
        void unsetInput(uint i);

        bool getOutput();
};

#endif
