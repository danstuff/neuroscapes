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

float Matrix::get(uint16 i, uint16 j){
    assert(i < depth && j < breadth);
    return data[i][j];
}

void Matrix::set(uint16 i, uint16 j, float value){
    assert(i < depth && j < breadth);
    data[i][j] = value;
}

void Matrix::add(uint16 i, uint16 j, float value){
    assert(i < depth && j < breadth);
    data[i][j] += value;
}

uint16 Matrix::getDepth(){
    return depth;
}

uint16 Matrix::getBreadth(){
    return breadth;
}

Matrix Matrix::copy(){
    return Matrix(data, depth, breadth);
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

Matrix Matrix::dot(Matrix& b){
    assert(getBreadth() == b.getDepth());
    
    Matrix ans(getDepth(), b.getBreadth());

    for(uint16 i = 0; i < getDepth(); i++){
        for(uint16 j = 0; j < b.getBreadth(); j++){
            ans.set(i, j, 0);

            for(uint16 k = 0; k < getBreadth(); k++){
                ans.add(i, j, get(i, k)*b.get(k,j));
            }
        }
    }

    return ans;
}
