#include "neunet.h"

uint16 NeuNet::getLyrBreadth(uint16 l){
    if(l == 1){ return NEUNET_INPUTS; }
    if(l == NEUNET_DEPTH-1){ return NEUNET_OUTPUTS; }
    return NEUNET_BREADTH;
}

NeuNet::NeuNet(){
    assert(NEUNET_INPUTS < NEUNET_BREADTH);
    assert(NEUNET_OUTPUTS < NEUNET_BREADTH);

    srand(RAND_SEED);

    //initialize all weights and biases to random #s
    for(uint16 l = 0; l < NEUNET_DEPTH; l++){
        cout << "Layer " << l << ":" << endl;

        for(uint16 i = 0; i < getLyrBreadth(l); i++){
            cout << "  Neuron " << i << ":" << endl;

            //input and output layers have no bias
            biases[l][i] = (l == 0 || l == NEUNET_DEPTH-1) ? 
                            0 : FtoN(randf(0.0f, 1.0f));
            
            cout << "   Bias: " << biases[l][i] <<  endl;
            cout << "   Weights: (";

            for(uint16 j = 0; j < getLyrBreadth(l-1); j++){
                //input and output layers have weight 1
                weights[l][i][j] = (l == 0 || l == NEUNET_DEPTH-1) ? 
                                    1 : FtoN(randf(0.0f, 1.0f));

                cout << weights[l][i][j] << ",";
            }

            cout << ")" << endl;
        }
    }
}

void NeuNet::feedfwd(float* a){
    //feeds the array a through every layer of the neural net
    Matrix am(a, NEUNET_BREADTH);

    //for each hidden layer
    for(uint16 l = 1; l < NEUNET_DEPTH; l++){

        //copy the last layer's activations into pm
        Matrix pm = am.copy();

        //convert the weights+biases for this layer into matricies
        Matrix wm(weights[l], getLyrBreadth(l), getLyrBreadth(l-1));
        Matrix bm(biases[l], getLyrBreadth(l));

        //next activations = sigmoid((old a's * weights) + biases)
        am = am.dot(wm).add(bm).sig();
    }
}

void NeuNet::backprop(float* a, float* y){
    //perform a feed forward but store all activiations and z values for each layer
    Matrix as[NEUNET_DEPTH];
    Matrix zs[NEUNET_DEPTH-1];

    Matrix ys(y, NEUNET_OUTPUTS);

    //0th layer activations are stored in a
    as[0] = Matrix(a, NEUNET_BREADTH);

    //for each non-input layer
    for(uint16 l = 1; l < NEUNET_DEPTH; l++){

        //convert the weights+biases for this layer into matricies
        Matrix wm(weights[l], getLyrBreadth(l), getLyrBreadth(l-1));
        Matrix bm(biases[l], getLyrBreadth(l));
        
        //calculate the z and activation values for this layer
        zs[l] = as[l-1].dot(wm).add(bm);
        as[l] = zs[l].sig();
    }

    //backward pass start
    //ABANDON ALL HOPE YE WHO ENTER HERE
    
    //calculate difference between y (desired activation) and a
    Matrix deltas = as[NEUNET_DEPTH-1].sub(ys);

    //nablas are the desired nudge to each weight or bias
    Matrix nabla_b[NEUNET_DEPTH];
    Matrix nabla_w[NEUNET_DEPTH];

    //last layer's nabla b is just the activation deltas
    nabla_b[NEUNET_DEPTH-1] = deltas.copy();

    //set nabla w to the dot product of act_tran and the deltas
    nabla_w[NEUNET_DEPTH-1] = deltas.dot(as[NEUNET_DEPTH-2].transpose());

    //for each layer, starting with the last hidden one
    for(uint16 l = NEUNET_DEPTH-2; l > 0; l--){

        //dot product trasnpose of weights of last layer with the deltas
        Matrix wm(weights[l+1], getLyrBreadth(l+1), getLyrBreadth(l));
        deltas = deltas.dot(wm.transpose());

        //also multiply by the sigmoid prime of the z values
        deltas = deltas.dot(zs[l].sigp());

        //the new deltas are your nabla b
        nabla_b[l] = deltas.copy();
    
        //set nabla w to the dot product of the a's transpose and the deltas
        nabla_w[l] = deltas.dot(as[l-1].transpose());
    }

    //TODO repeat for many trials, then apply to the weights and biases
}

