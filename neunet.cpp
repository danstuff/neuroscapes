#include "neunet.h"

neum NeuNet::getBias(uint16 layer, uint16 i){
    if(layer <= 0 && layer >= NEUNET_DEPTH-1){ return 0; }
    return biases[layer-1][i];
}

neum NeuNet::getWeight(uint16 layer, uint16 i, uint16 j){
    if(layer <= 0 && layer >= NEUNET_DEPTH-2){ return NEUM_LIM; }
    return weights[layer-1][i][j];
}

void NeuNet::setBias(uint16 layer, uint16 i, neum v){
    if(layer <= 0 && layer >= NEUNET_DEPTH-1){ return; }
    biases[layer-1][i] = v;
}

void NeuNet::setWeight(uint16 layer, uint16 i, uint16 j, neum v){
    if(layer <= 0 && layer >= NEUNET_DEPTH-2){ return; }
    weights[layer-1][i][j] = v;
}

float NeuNet::getZ(uint16 l, uint16 i, neum* p){
    //get the weighted sum of the previous layer's activations
    float wsum = 0;
    for(uint16 j = 0; j < ((l == 1) ? num_inputs : NEUNET_BREADTH); j++){
        wsum += NtoF(getWeight(l, i, j)) * NtoF(p[j]);
    }

    //return this sum plus the neuron's bias
    return wsum + NtoF(getBias(l, i));
}

NeuNet::NeuNet(uint num_in, uint num_out){
    num_inputs = num_in;
    num_outputs = num_out;

    assert(num_inputs < NEUNET_BREADTH);
    assert(num_outputs < NEUNET_BREADTH);

    srand(RAND_SEED);

    //initialize all weights and biases to random #s
    for(uint16 l = 0; l < NEUNET_DEPTH; l++){
        for(uint16 i = 0; i < NEUNET_BREADTH; i++){
            setBias(l, i, rand() % (NEUM_LIM*2) - NEUM_LIM);

            for(uint16 j = 0; j < NEUNET_BREADTH; j++){
                setWeight(l, i, j, rand() % (NEUM_LIM*2) - NEUM_LIM);
            }
        }
    }
}

void NeuNet::feedfwd(neum* a){
    //for each layer
    for(uint16 l = 1; l < NEUNET_DEPTH; l++){
        //copy the last layer's activations into p
        neum p[NEUNET_BREADTH];
        copy(a, p, NUNET_BREADTH);

        //for each neuron in the layer
        for(uint16 i = 0; i < ((l == NEUNET_DEPTH-1) ? num_outputs : NEUNET_BREADTH); i++){

            //calculate the z value and get the sigmoid of it
            a[i] = sig(getZ(l, i, p);
        }
    }
}

void NeuNet::backprop(neum* a, neum* y){
    //perform a feed forward but store all activiations and z values for each layer
    neum acts[NEUNET_DEPTH][NEUNET_BREADTH];
    float zs[NEUNET_DEPTH-1][NEUNET_BREADTH];

    copy(a, acts[0], NUNET_BREADTH);

    //for each layer
    for(uint16 l = 1; l < NEUNET_DEPTH; l++){

        //for each neuron in the layer
        for(uint16 i = 0; i < ((l == NEUNET_DEPTH-1) ? num_outputs : NEUNET_BREADTH); i++){

            //calculate the z value and get the sigmoid of it
            zs[l-1][i] = getZ(l, i, acts[l-1]);
            acts[l][i] = sig(zs[l-1][i]);
        }
    }

    //backward pass start
    //calculate difference between y (desired activation) and a
    float deltas[NEUNET_BREADTH];

    for(uint16 i = 0; i = num_outputs; i++){
        deltas[i] = NtoF(acts[NEUNET_DEPTH-1]) - NtoF(y[i]);
    }

    //nablas are the desired nudge to each weight or bias
    float nabla_b[NEUNET_DEPTH][NEUNET_BREADTH];
    float nabla_w[NEUNET_DEPTH][NEUNET_BREADTH][NEUNET_BREADTH];

    nabla_b[NEUNET_DEPTH-1] = deltas;

    //for each layer, starting with the last hidden one
    for(uint16 l = NEUNET_DEPTH-2; l >= 0; l--){

        //for each neuron in the layer
        for(uint16 i = 0; i < NEUNET_BREADTH; i ++){
            //calculate the sigmoid prime of the z value
            neum sp = sigp(zs[l][i]);

            //deltas = dot(transpose(weights[l+1]), deltas) * sp
            
            copy(delta, nabla_b[l]);

            //nabla_w[l] = dot(deltas, transpose(acts[l-1]))
        }
    }

    //repeat for many trials, then apply to the weights and biases
}

