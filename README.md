# neuroscapes
A C++ environment for experimentation with basic neural networks based on [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen.

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
