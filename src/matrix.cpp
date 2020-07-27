#include "util.h"

Matrix::Matrix(uint16 d, uint16 b){
    assert(d <= MAT_SIZE && b <= MAT_SIZE);

    depth = d;
    breadth = b;

    for(uint16 i = 0; i < d; i++){
        for(uint16 j = 0; j < b; j++){
            data[i][j] = 0;
        }
    }
}

Matrix::Matrix(float** orig_data, uint16 d, uint16 b){
    assert(d <= MAT_SIZE && b <= MAT_SIZE);

    depth = d;
    breadth = b;

    for(uint16 i = 0; i < d; i++){
        for(uint16 j = 0; j < b; j++){
            data[i][j] = orig_data[i][j];
        }
    }
}

Matrix::Matrix(float* orig_data, uint16 d){
    assert(d <= MAT_SIZE);

    depth = d;
    breadth = 1;

    for(uint16 i = 0; i < d; i++){
        data[i][1] = orig_data[i];
    }
}

uint16 Matrix::getDepth(){
    return depth;
}

uint16 Matrix::getBreadth(){
    return breadth;
}

float** Matrix::getData(){
    return data;
}

Matrix Matrix::copy(){
    return Matrix(data, getDepth(), getBreadth());
}

Matrix Matrix::transpose(){
    Matrix ans(getBreadth(), getDepth());

    for(uint16 i = 0; i < getDepth(); i++){
        for(uint16 j = 0; j < getBreadth(); j++){
            //swap i,j and j,i in the answer
            ans.set(j, i, get(i, j));
        }
    }

    return ans;
}

Matrix Matrix::add(Matrix& b){
    assert(getDepth() == b.getDepth() &&
            getBreadth() == b.getBreadth());

    Matrix ans(getDepth(), getBreadth());

    for(uint16 i = 0; i < getDepth(); i++){
        for(uint16 j = 0; j < getBreadth(); j++){
            ans.getData()[i][j] = data[i][j] + b[i][j];
        }
    }

    return ans;
}

Matrix Matrix::sub(Matrix& b){
    assert(getDepth() == b.getDepth() &&
            getBreadth() == b.getBreadth());

    Matrix ans(getDepth(), getBreadth());

    for(uint16 i = 0; i < getDepth(); i++){
        for(uint16 j = 0; j < getBreadth(); j++){
            ans.getData()[i][j] = data[i][j] - b[i][j];
        }
    }

    return ans;
}

Matrix Matrix::dot(Matrix& b){
    assert(getBreadth() == b.getDepth());
    
    Matrix ans(getDepth(), b.getBreadth());

    for(uint16 i = 0; i < getDepth(); i++){
        for(uint16 j = 0; j < b.getBreadth(); j++){
            ans.getData()[i][j] = 0;

            for(uint16 k = 0; k < getBreadth(); k++){
                ans.getData()[i][j] += data[i][k]*b.getData()[k][j];
            }
        }
    }

    return ans;
}

Matrix Matrix::sig(){
    Matrix ans(getDepth(), getBreadth());

    for(uint16 i = 0; i < getDepth(); i++){
        for(uint16 j = 0; j < getBreadth(); j++){
            ans.getData()[i][j] = sig(ans.getData()[i][j]);
        }
    }

    return ans;
}

Matrix Matrix::sigp(){
    Matrix ans(getDepth(), getBreadth());

    for(uint16 i = 0; i < getDepth(); i++){
        for(uint16 j = 0; j < getBreadth(); j++){
            ans.getData()[i][j] = sigp(ans.getData()[i][j]);
        }
    }

    return ans;
}
