#include "include/neunet.h"

NeuNet neunet;

float zer[] = { 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
                0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
                0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
                0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
                0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
                0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
                0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
                0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f };

float one[] = { 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
                1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
                1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
                1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
                1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
                1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
                1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
                1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f };

int main(){
    //feed a matrix of alternating bits through the neural network and print the result
    Matrix a0(zer, NEUNET_INPUTS); 
    neunet.feedfwd(a0);
    a0.trunc().print();

    //train the neural network 200 times with alternating bits
    for(uint16 i = 0; i < 100; i++){
        Matrix a[2];
        a[0] = Matrix(zer, NEUNET_INPUTS);
        a[1] = Matrix(one, NEUNET_INPUTS);

        Matrix y[2];
        y[0] = Matrix(one, NEUNET_OUTPUTS);
        y[1] = Matrix(zer, NEUNET_OUTPUTS);

        neunet.backprop(a, y, 2, 200, 1, 0.001);
    }
    
    //feed a matrix of alternating bits through the neural network and print the result
    Matrix a1(zer, NEUNET_INPUTS); 
    neunet.feedfwd(a1);
    a1.trunc().print();
    
    //write the final neural network to a file
    neunet.write("test.dat");
    
    return 0;
}
