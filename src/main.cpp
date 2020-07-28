#include "include/neunet.h"

int main(){
    NeuNet neunet;

    //feed some test values into the net
    float af[] = { 1.0f, 0.0f, 1.0f };
    Matrix a(af, 3); 

    neunet.feedfwd(a);

    //print result of feed forward
    cout << "Result: ";

    a.print();
    
    //wait forever
    while(true);
}
