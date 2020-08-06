#include "include/neunet.h"

uint16 NeuNet::getLyrBreadth(uint16 l){
    if(l == 0){ return NEUNET_INPUTS; }
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
                            0 : randf(-1.0f, 1.0f);
            
            for(uint16 j = 0; j < getLyrBreadth(l-1); j++){
                //input and output layers have weight 1
                weights[l][i][j] = (l == 0 ) ? 
                                    1 : randf(-1.0f, 1.0f);
            }
        }
    }
}

void NeuNet::print(){
    for(uint16 l = 0; l < NEUNET_DEPTH; l++){
        cout << "Layer " << l << ":" << endl;

        for(uint16 i = 0; i < getLyrBreadth(l); i++){
            cout << "  Neuron " << i << ":" << endl;
            cout << "   Bias: " << biases[l][i] <<  endl;
            cout << "   Weights: (";

            for(uint16 j = 0; j < getLyrBreadth(l-1); j++){
                cout << weights[l][i][j] << ",";
            }

            cout << ")" << endl;
        }
    }
}

void NeuNet::write(const char* filename){
    ofstream  file(filename, ios::out | ios::binary);

    //for every layer
    for(uint16 l = 1; l < NEUNET_DEPTH; l++){

        //for every neuron
        for(uint16 i = 0; i < getLyrBreadth(l); i++){

            //output the bias
            file.write((char*)&biases[l][i], 4);
            
            //for every weight
            for(uint16 j = 0; j < getLyrBreadth(l-1); j++){

                //output the weight
                file.write((char*)&weights[l][i][j], 4);
            }
        }
    }

    file.close();
}

void NeuNet::read(const char* filename){
    ifstream  file(filename, ios::in | ios::binary);

    //ensure file actually exists
    if(!file.is_open()){ return; }

    //for every layer
    for(uint16 l = 1; l < NEUNET_DEPTH; l++){

        //for every neuron
        for(uint16 i = 0; i < getLyrBreadth(l); i++){

            //read the bias
            file.read((char*)&biases[l][i], 4);
            
            //for every weight
            for(uint16 j = 0; j < getLyrBreadth(l-1); j++){

                //read the weight
                file.read((char*)&weights[l][i][j], 4);
            }
        }
    }

    file.close();
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

        //confirm that the depth of a is same as the layer breadth
        a.depth = getLyrBreadth(l);
        
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
        const uint16 last = NEUNET_DEPTH-1;

        //calculate difference between y (desired activation) and a
        Matrix deltas = as[last].sub(trial_ys[trial]);

        //nablas are the desired nudge to each weight or bias
        Matrix d_nabla_b[NEUNET_DEPTH];
        Matrix d_nabla_w[NEUNET_DEPTH];

        //last layer's nabla b is just the activation deltas
        d_nabla_b[last] = deltas.copy();

        //set nabla w to the dot product of the prev layer's activation transpose and the deltas
        d_nabla_w[last] = deltas.dot(as[last-1].transpose());

        //apply delta nablas to overall nabla
        nabla_b[last] = nabla_b[last].add(d_nabla_b[last]);
        nabla_w[last] = nabla_w[last].add(d_nabla_w[last]);

        //for each layer, starting with the last hidden one
        for(uint16 l = last-1; l > 0; l--){
            
            //set deltas to transpose of weights . deltas * sigp(zs)
            Matrix sp = zs[l].sigp();
            Matrix w(weights[l+1], getLyrBreadth(l+1), getLyrBreadth(l));

            deltas = w.transpose().dot(deltas).mul(sp);

            //the new deltas are your nabla b
            d_nabla_b[l] = deltas.copy();
        
            //set nabla w to the dot product of the a's transpose and the deltas
            d_nabla_w[l] = deltas.dot(as[l-1].transpose());

            //apply delta nablas to overall nabla
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
            float b = biases[l][i];
            float nb = nabla_b[l].data[i][0];

            b = b - (learn_rate/(float)num_trials)*nb;
           
            biases[l][i] = b;

            //for each weight of the neuron
            for(uint16 j = 0; j < getLyrBreadth(l-1); j++){

                //adjust the weight based on nabla w
                float w = weights[l][i][j];
                float nw = nabla_w[l].data[i][j];

                //(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                w = (1-learn_rate*(reg/(float)total_size))*w -
                    (learn_rate/(float)num_trials)*nw;

                weights[l][i][j] = w;
            }
        }
    }
}

