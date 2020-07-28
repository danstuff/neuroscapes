#include "include/neunet.h"

uint16 NeuNet::getLyrBreadth(uint16 l){
    if(l == 1){ return NEUNET_INPUTS; }
    if(l == NEUNET_DEPTH-1){ return NEUNET_OUTPUTS; }
    return NEUNET_BREADTH;
}

NeuNet::NeuNet(){
    assert(NEUNET_INPUTS <= NEUNET_BREADTH);
    assert(NEUNET_OUTPUTS <= NEUNET_BREADTH);

    srand(RAND_SEED);

    //initialize all weights and biases to random #s
    for(uint16 l = 0; l < NEUNET_DEPTH; l++){
        for(uint16 i = 0; i < getLyrBreadth(l); i++){

            //input and output layers have no bias
            biases[l][i] = (l == 0) ? 
                            0 : FtoN(randf(0.0f, 1.0f));
            
            for(uint16 j = 0; j < getLyrBreadth(l-1); j++){
                //input and output layers have weight 1
                weights[l][i][j] = (l == 0 ) ? 
                                    FtoN(1) : FtoN(randf(0.0f, 1.0f));
            }
        }
    }
}

void NeuNet::print(){
    for(uint16 l = 0; l < NEUNET_DEPTH; l++){
        cout << "Layer " << l << ":" << endl;

        for(uint16 i = 0; i < getLyrBreadth(l); i++){
            cout << "  Neuron " << i << ":" << endl;
            cout << "   Bias: " << NtoF(biases[l][i]) <<  endl;
            cout << "   Weights: (";

            for(uint16 j = 0; j < getLyrBreadth(l-1); j++){
                cout << NtoF(weights[l][i][j]) << ",";
            }

            cout << ")" << endl;
        }
    }
}

void NeuNet::feedfwd(Matrix& a, Matrix* a_collect, Matrix* z_collect){
    //feeds the array a through every layer of the neural net
    if(a_collect != NULL){
        a_collect[0] = a.copy();
    }

    //for each hidden layer
    for(uint16 l = 1; l < NEUNET_DEPTH; l++){

        //convert the weights+biases for this layer into matricies
        Matrix w(weights[l], getLyrBreadth(l), getLyrBreadth(l-1));
        Matrix b(biases[l], getLyrBreadth(l));

        //next activations = sigmoid((old a's * weights) + biases)
        Matrix z = w.dot(a).add(b);
        a = z.sig();
        
        if(a_collect != NULL){
            a_collect[l] = a.copy();
        }

        if(z_collect != NULL){
            z_collect[l] = z.copy();
        }
    }
}

void NeuNet::backprop(Matrix* trial_as, Matrix* trial_ys, 
                      uint16 num_trials, uint16 total_size, 
                      float learn_rate, float reg){
    //ABANDON ALL HOPE YE WHO ENTER HERE
    Matrix nabla_b[NEUNET_DEPTH];
    Matrix nabla_w[NEUNET_DEPTH];

    for(uint16 l = 1; l < NEUNET_DEPTH; l++){
        nabla_b[l] = Matrix(getLyrBreadth(l), 1);
        nabla_w[l] = Matrix(getLyrBreadth(l), getLyrBreadth(l-1));
    }

    //for each trial
    for(uint16 trial = 0; trial < num_trials; trial++){

        //perform a feed forward and store activiations and z values for each layer
        Matrix as[NEUNET_DEPTH];
        Matrix zs[NEUNET_DEPTH];

        feedfwd(trial_as[trial], as, zs);
        
        //begin backward pass
        //calculate difference between y (desired activation) and a
        Matrix deltas = as[NEUNET_DEPTH-1].sub(trial_ys[trial]);
        deltas.print();

        //nablas are the desired nudge to each weight or bias
        Matrix d_nabla_b[NEUNET_DEPTH];
        Matrix d_nabla_w[NEUNET_DEPTH];

        //last layer's nabla b is just the activation deltas
        d_nabla_b[NEUNET_DEPTH-1] = deltas.copy();

        //set nabla w to the dot product of act_tran and the deltas
        d_nabla_w[NEUNET_DEPTH-1] = deltas.dot(as[NEUNET_DEPTH-2].transpose());

        //for each layer, starting with the last hidden one
        for(uint16 l = NEUNET_DEPTH-2; l > 0; l--){

            //dot product trasnpose of weights of next layer with the deltas
            Matrix w(weights[l+1], getLyrBreadth(l+1), getLyrBreadth(l));
            deltas = w.transpose().dot(deltas);

            //also multiply by the sigmoid prime of the z values
            deltas = deltas.mul(zs[l].sigp());

            //the new deltas are your nabla b
            d_nabla_b[l] = deltas.copy();
        
            //set nabla w to the dot product of the a's transpose and the deltas
            d_nabla_w[l] = deltas.dot(as[l-1].transpose());

            nabla_b[l] = nabla_b[l].add(d_nabla_b[l]);
            nabla_w[l] = nabla_w[l].add(d_nabla_w[l]);
        }
    }

    //apply to the weights and biases
    //for each hidden layer
    for(uint16 l = 1; l < NEUNET_DEPTH; l++){

        //for each neuron in the layer
        for(uint16 i = 0; i < getLyrBreadth(l); i++){

            //adjust the bias based on nabla b
            float b = NtoF(biases[l][i]);
            float nb = nabla_b[l].data[i][0];

            b = b - ((float)learn_rate/(float)num_trials)*nb;
           
            biases[l][i] = FtoN(b);

            //for each weight of the neuron
            for(uint16 j = 0; j < getLyrBreadth(l-1); j++){

                //adjust the weight based on nabla w
                float w = NtoF(weights[l][i][j]);
                float nw = nabla_w[l].data[i][j];

                w = w + (nw - w)/learn_rate;

                weights[l][i][j] = FtoN(w);
            }
        }
    }
}

