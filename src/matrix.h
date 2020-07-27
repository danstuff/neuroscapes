#pragma once

#ifndef MATRIX_H
#define MATRIX_H

#include "util.h"

const uint16 MAT_SIZE = 16;

class Matrix{
    private:
        float data[MAT_SIZE][MAT_SIZE];
        uint16 depth, breadth;

    public:
        Matrix(){};

        Matrix(uint16 d, uint16 b);
        
        Matrix(float** orig_data, uint16 d, uint16 b);
        Matrix(float* orig_data, uint16 d);

        Matrix(neum** orig_data, uint16 d, uint16 b);
        Matrix(neum* orig_data, uint16 d);

        uint16 getDepth();
        uint16 getBreadth();

        float** getData();

        Matrix copy();

        Matrix transpose();
        
        Matrix add(Matrix b);
        Matrix sub(Matrix b);

        Matrix dot(Matrix b);

        Matrix sig();
        Matrix sigp();
};

#endif
