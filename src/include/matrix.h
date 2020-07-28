#pragma once

#ifndef MATRIX_H
#define MATRIX_H

#include "include/util.h"

const uint16 MAT_SIZE = 4;
const uint16 MAT_ACCEPT = 1;

class Matrix{
    public:
        float data[MAT_SIZE][MAT_SIZE];
        uint16 depth, breadth;

        Matrix(){};
        Matrix(uint16 d, uint16 b);
        
        Matrix(neum orig_data[MAT_ACCEPT][MAT_ACCEPT], uint16 d, uint16 b);
        Matrix(neum* orig_data, uint16 d);

        Matrix(float orig_data[MAT_SIZE][MAT_SIZE], uint16 d, uint16 b);
        Matrix(float* orig_data, uint16 d);

        void print();

        Matrix copy();

        Matrix transpose();
        
        Matrix add(Matrix b);
        Matrix sub(Matrix b);

        Matrix dot(Matrix b);
        Matrix mul(Matrix b);

        Matrix sig();
        Matrix sigp();
};

#endif
