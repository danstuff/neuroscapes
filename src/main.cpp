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

void feed(){
    //neunet.print();

     //feed some test values into the net
    Matrix a(zer, NEUNET_INPUTS); 
    neunet.feedfwd(a);

    //print result of feed forward
    cout << "Result (zer): ";
    a.trunc().print();

     //feed some test values into the net
    Matrix b(one, NEUNET_INPUTS); 
    neunet.feedfwd(b);

    //print result of feed forward
    cout << "Result (one): ";
    b.trunc().print();

    cout << endl << endl;
}

int main(){

    feed();

    for(uint16 i = 0; i < 100; i++){
        Matrix a[2];
        a[0] = Matrix(zer, NEUNET_INPUTS);
        a[1] = Matrix(one, NEUNET_INPUTS);

        Matrix y[2];
        y[0] = Matrix(one, NEUNET_OUTPUTS);
        y[1] = Matrix(zer, NEUNET_OUTPUTS);

        neunet.backprop(a, y, 2, 200, 1, 0.001);
    }
    
    feed();
   
    //wait forever
    while(true);
}
