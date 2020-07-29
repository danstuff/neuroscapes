#include "include/neunet.h"

NeuNet neunet;

void feed(){
    neunet.print();

     //feed some test values into the net
    float arr[] = { 0.0f };
    Matrix a(arr, 1); 

    neunet.feedfwd(a);

    //print result of feed forward
    cout << "Result: ";
    a.print();

}

int main(){

    feed();

    cout << endl;

    for(uint16 i = 0; i < 1000; i++){
        float af[] = { 0.0f };
        Matrix a(af, 1);

        float yf[] = { 0.5f };
        Matrix y(yf, 1);

        neunet.backprop(&a, &y, 1, 1, 0.5, 0.005);
    }
    
    cout << endl;
    
    feed();
   
    //wait forever
    while(true);
}
