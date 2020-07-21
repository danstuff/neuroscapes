#include "neunet.h"

NeuNet::NeuNet(uint num_in, uint num_out){
    num_inputs = num_in;
    num_outputs = num_out;

    assert(num_inputs < NEU_MAX_CON);
    assert(num_outputs < NEU_MAX_CON);

    //construct first layer of neurons (input layer)
    for(uint i = 1; i < num_inputs; i++){
        layers[0][i] = Neuron(1);
    }

    //construct first hidden layer of neurons
    for(uint i = 0; i < NEU_MAX_CON; i++){
        layers[1][i] = Neuron(num_inputs);
    }

    //construct remaining hidden layers
    for(uint i = 1; i < NEUNET_MAX_LAYER-1; i++){
        for(uint j = 0; j < NEU_MAX_CON; j++){
            layers[i][j] = Neuron(NEU_MAX_CON);
        }
    }

    //construct last layer of neurons (output layer)
    for(uint i = 1; i < num_outputs; i++){
        layers[NEUNET_MAX_LAYER-1][i] = Neuron(NEU_MAX_CON);
    }
}

void NeuNet::randomizeWeights(){
     //randomize input layer
    for(uint i = 0; i < num_inputs; i++){
        layers[0][i].randomizeWeights();
    }

    //randomize hidden layers
    for(uint l = 1; l < NEUNET_NUM_LAYERS-1; l++){
        for(uint i = 0; i < NEU_MAX_CON; i++){
            layers[l][i].randomizeWeights();
        }
    }

    //randomize output layer
    for(uint i = 0; i < num_outputs; i++){
        layers[NEUNET_NUM_LAYERS-1][i].randomizeWeights();
    }

}

uint getNumInputs(){
    return num_inputs;
}

uint getNumOutputs(){
    return num_outputs;
}

void run(uint* inputs){
    //set inputs to the first layer to the values in the input array
    for(uint i = 0; i < num_inputs; i++){
        layers[0][i].setInput(0, inputs[i]);
    }

    //set inputs on the first hidden layer to outputs of the first layer
    for(uint i = 0; i < NEU_MAX_CON; i++){
        for(uint j = 0; j < num_inputs; j++){
            layers[1][i].setInput(0, layers[0][j].getOutput());
        }
    }

    //for each neuron on each hidden layer
    for(uint l = 2; l < NEUNET_NUM_LAYERS-1; l++){
        for(uint i = 0; i < NEU_MAX_CON; i++){

            //make all the outputs of the previous layer inputs on this one 
            for(uint j = 0; j < NEU_MAX_CON; j++){
                layers[l][i].setInput(j, layers[l-1][j].getOutput());
            }
        }
    }

    //for each neuron on the output layer
    for(uint i = 0; i < num_outputs; i++){

        //set the inputs to be all the outputs of the previous layer
        for(uint j = 0; j < NEU_MAX_CON; j++){
            uint o = layers[NEUNET_NUM_LAYERS-2][j].getOutput();
            layers[NEUNET_NUM_LAYERS-1][i].setInput(j, o);
        }
    }
}

uint getOutput(uint i){
    assert(i < num_outputs);
    return layers[NEUNET_NUM_LAYERS-1][i].getOutput();
}

void backPropagate(uint* expected){
    //TODO figure out how the fuck this works

}

