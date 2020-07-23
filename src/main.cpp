#include "neunet.h"

int main(){
    NeuNet neunet(3, 3);

    //feed some test values into the net
    float a[3] = { 1, 0, 1 }; 

    neunet.feedfwd(a);

    //print result of feed forward
    cout << "Result: ";

    for(uint16 i = 0; i < 3; i++){
        cout << a[i] << ", ";
    }

    cout << endl;
    
    //wait forever
    while(true);
}
