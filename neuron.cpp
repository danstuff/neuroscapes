#include "neuron.h"

Neuron::Neuron(uint num_ipts, float ibias){
    num_inputs = num_ipts;
    bias = ibias;
    
    assert(num_inputs <= NEU_MAX_CON); 

    clearInputs();
}

void Neuron::clearInputs(){
    for(uint i = 0; i < num_inputs; i++){
        inputs[i].value = 0;
        inputs[i].weight = 1;
    }
}

void Neuron::randomizeWeights(){
    for(uint i = 0; i < num_inputs; i++){
        inputs[i].weight = 1; //TODO
    }
}

void Neuron::setInput(uint i, uint value){
    assert(i < num_inputs); 
    inputs[i].value = value;
}

void setInputWeight(uint i, float weight){
    assert(i < num_inputs); 
    inputs[i].weight = weight;
}

bool Neuron::getOutput(){
    //get a weighted sum of all enabled inputs
    float output = bias;

    for(uint i = 0; i < num_inputs; i++){
        output += inputs[i].value * inputs[i].weight;
    }

    //run output through a basic linear sigmoid function
    output = (2 / (1 + pow(5, output)) - 1;
    
    return output;
}
