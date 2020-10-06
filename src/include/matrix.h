#pragma once

#ifndef MATRIX_H
#define MATRIX_H

#include "util.h"

class Matrix{
    public:
        vector<vector<float>> data;
        uint16 depth, breadth;

        Matrix(){};
        Matrix(uint16 d, uint16 b);

        void patternFill(float a, float b);

        void setRow(uint16 row, float* orig_data);
        
        void print();

        Matrix copy();

        Matrix transpose();
        
        Matrix add(Matrix b);
        Matrix sub(Matrix b);

        Matrix dot(Matrix b);
        Matrix mul(Matrix b);

        Matrix sig();
        Matrix sigp();


        Matrix trunc();
};

#endif
