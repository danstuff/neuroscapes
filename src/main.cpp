#include "include/neunet.h"

NeuNet neunet;

float zer[] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
       		0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	       	0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	       	0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

float one[] = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
       		0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	       	0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	       	0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

void feed(){
    neunet.print();

     //feed some test values into the net
    Matrix a(zer, 32); 
    neunet.feedfwd(a);

    //print result of feed forward
    cout << "Result (a=0): ";
    a.print();

     //feed some test values into the net
    Matrix b(one, 2); 
    neunet.feedfwd(b);

    //print result of feed forward
    cout << "Result (a=1): ";
    b.print();

    cout << endl;
}

int main(){

    feed();

    for(uint16 i = 0; i < 1000; i++){
        Matrix a[2];
        a[0] = Matrix(zer, 1);
        a[1] = Matrix(one, 1);

        Matrix y[2];
        y[0] = Matrix(one, 1);
        y[1] = Matrix(zer, 1);

        neunet.backprop(a, y, 2, 2, 5, 0.0001);
    }
    
    feed();
   
    //wait forever
    while(true);
}
