# neuroscapes
A multi-platform environment for experimentation with basic neural networks based on [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen.

## Installation and Use
### Compiling from source
In order to use neuroscapes, you will have to compile it from source. This requires you to either have a linux machine with `make`  and `gcc` installed or a Windows machine with a GNU alternative installed like MinGW or cygwin. First, open a console window and navigate to a directory of your choice. Then use `git clone` to clone the repository to your computer like so:

    git clone https://github.com/danstuff/neuroscapes.git

To compile the source code, enter the newly created neuroscapes directory, create a new directory for the DLL output called `bin`, then run `make`:

    cd neuroscapes
    mkdir bin
    make
    
This will generate a DLL file that you can link to a C++ project.

### Linking
To link a C++ project with the neuroscapes library, you will need to put its header files and your compiled DLL in a place where your project can find them. You can find the header files under `src/include` in this repository. The linking process will vary depending on the compiler you use; I recommend reading your compiler's official documentation for more information on this. If you are using a Windows machine with MinGW, copy the DLL to two places: put it in the same directory as your project executeable, and in the `lib` folder in MinGW's install directory. Next, go to the `include` folder in MinGW's install directory and create a new subfolder called `neuroscapes`. Copy all the `.h` header files from `src/include` into this new subfolder.

### Example code
The below code uses the neuroscapes library to create a neural network that acts as a simple bit flipper.
```c++
#include "neuroscapes/neunet.h"

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
```
