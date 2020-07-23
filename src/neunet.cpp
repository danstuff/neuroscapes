#include "neunet.h"

float NeuNet::getBias(uint16 layer, uint16 i){
    if(layer <= 0 && layer >= NEUNET_DEPTH-1){ return 0; }
    return NtoF(biases[layer-1][i]);
}

float NeuNet::getWeight(uint16 layer, uint16 i, uint16 j){
    if(layer <= 0 && layer >= NEUNET_DEPTH-2){ return NEUM_LIM; }
    return NtoF(weights[layer-1][i][j]);
}

void NeuNet::setBias(uint16 layer, uint16 i, float v){
    if(layer <= 0 && layer >= NEUNET_DEPTH-1){ return; }
    biases[layer-1][i] = FtoN(v);
}

void NeuNet::setWeight(uint16 layer, uint16 i, uint16 j, float v){
    if(layer <= 0 && layer >= NEUNET_DEPTH-2){ return; }
    weights[layer-1][i][j] = FtoN(v);
}

uint16 NeuNet::getLyrBreadth(uint16 l){
    if(l == 1){ return num_inputs; }
    if(l == NEUNET_DEPTH-1){ return num_outputs; }
    return NEUNET_BREADTH;
}

float NeuNet::getZ(uint16 l, uint16 i, float* p){
    //get the weighted sum of the previous layer's activations
    float wsum = 0;
    for(uint16 j = 0; j < getLyrBreadth(l-1); j++){
        wsum += getWeight(l, i, j) * p[j];
    }

    //return this sum plus the neuron's bias
    return wsum + getBias(l, i);
}

NeuNet::NeuNet(uint16 num_in, uint16 num_out){
    num_inputs = num_in;
    num_outputs = num_out;

    assert(num_inputs < NEUNET_BREADTH);
    assert(num_outputs < NEUNET_BREADTH);

    srand(RAND_SEED);

    //initialize all weights and biases to random #s
    for(uint16 l = 0; l < NEUNET_DEPTH; l++){
        cout << "Layer " << l << ":" << endl;

        for(uint16 i = 0; i < NEUNET_BREADTH; i++){
            cout << "  Neuron " << i << ":" << endl;

            setBias(l, i, randf(0.0f, 1.0f));
            
            cout << "   Bias: " << getBias(l, i) <<  endl;
            cout << "   Weights: (";

            for(uint16 j = 0; j < NEUNET_BREADTH; j++){
                setWeight(l, i, j, randf(0.0f, 1.0f));

                cout << getWeight(l, i, j) << ",";
            }

            cout << ")" << endl;
        }
    }
}

void NeuNet::feedfwd(float* a){
    //feeds the array a through every layer of the neural net

    //for each layer
    for(uint16 l = 1; l < NEUNET_DEPTH; l++){
        //copy the last layer's activations into p
        float p[NEUNET_BREADTH];
        copy(a, p, NEUNET_BREADTH);

        //for each neuron in the layer
        for(uint16 i = 0; i < getLyrBreadth(l); i++){

            //calculate the z value and get the sigmoid of it
            a[i] = sig(getZ(l, i, p));
        }
    }
}

void NeuNet::backprop(float* a, float* y){
    //perform a feed forward but store all activiations and z values for each layer
    float acts[NEUNET_DEPTH][NEUNET_BREADTH];
    float zs[NEUNET_DEPTH-1][NEUNET_BREADTH];

    copy(a, acts[0], NEUNET_BREADTH);

    //for each layer
    for(uint16 l = 1; l < NEUNET_DEPTH; l++){

        //for each neuron in the layer
        for(uint16 i = 0; i < getLyrBreadth(l); i++){

            //calculate the z value and get the sigmoid of it
            zs[l-1][i] = getZ(l, i, acts[l-1]);
            acts[l][i] = sig(zs[l-1][i]);
        }
    }

    //backward pass start
    //ABANDON ALL HOPE YE WHO ENTER HERE
    
    //calculate difference between y (desired activation) and a
    float deltas[NEUNET_BREADTH];

    for(uint16 i = 0; i = num_outputs; i++){
        deltas[i] = acts[NEUNET_DEPTH-1][i] - y[i];
    }

    //nablas are the desired nudge to each weight or bias
    float nabla_b[NEUNET_DEPTH][NEUNET_BREADTH];
    float nabla_w[NEUNET_DEPTH][NEUNET_BREADTH][NEUNET_BREADTH];

    //last layer's nabla b is just the activation deltas
    copy(deltas, nabla_b[NEUNET_DEPTH-1], NEUNET_BREADTH);

    //get the transpose of the last hidden layer's activations
    float act_tran[1][NEUNET_BREADTH];
    transpose(acts[NEUNET_DEPTH-2], act_tran, NEUNET_BREADTH, 1);
    
    //set nabla w to the dot product of act_tran and the deltas
    dot(deltas, act_tran, nabla_w, NEUNET_BREADTH, 1, 1, NEUNET_BREADTH);

    //for each layer, starting with the last hidden one
    for(uint16 l = NEUNET_DEPTH-2; l > 0; l--){

        //get the transpose of the weights for this layer
        float wei_tran[NEUNET_BREADTH][NEUNET_BREADTH];
        transpose(weights[l+1], wei_tran, NEUNET_BREADTH, NEUNET_BREADTH);

        //dot product trasnpose of weights with the deltas
        float nd[NEUNET_BREADTH];
        dot(wei_tran, deltas, nd, NEUNET_BREADTH, NEUNET_BREADTH, NEUNET_BREADTH, 1); 
        copy(nd, deltas, NEUNET_BREADTH);

        //multiply each delta by the corresponding sig prime of z
        for(uint16 i = 0; i < NEUNET_BREADTH; i++){
            deltas[i] *= sigp(zs[l][i]);
        }

        //the new deltas are your nabla b
        copy(deltas, nabla_b[l], NEUNET_BREADTH);

        //get the transpose of the last hidden layer's activations
        float act_tran[1][NEUNET_BREADTH];
        transpose(acts[l-1], act_tran, NEUNET_BREADTH, 1);
    
        //set nabla w to the dot product of act_tran and the deltas
        dot(deltas, act_tran, nabla_w, NEUNET_BREADTH, 1, 1, NEUNET_BREADTH);
    }

    //TODO repeat for many trials, then apply to the weights and biases
}

