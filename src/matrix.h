#pragma once

#ifndef MATRIX_H
#define MATRIX_H

#include "util.h"

const uint16 MAT_SIZE = 256;

class Matrix{
    private:
        float data[MAT_SIZE][MAT_SIZE];
        uint16 depth, breadth;

    public:
        Matrix(uint16 d, uint16 b);
        Matrix(float** orig_data, uint16 d, uint16 b);
        Matrix(float* orig_data, uint16 d);

        float get(uint16 i, uint16 j);

        void set(uint16 i, uint16 j, float value);
        void add(uint16 i, uint16 j, float value);

        uint16 getDepth();
        uint16 getBreadth();

        Matrix copy();
        Matrix transpose();
        Matrix dot(Matrix& b);
}

#endif
