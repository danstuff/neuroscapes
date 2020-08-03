#include "include/matrix.h"

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

Matrix::Matrix(float* orig_data, uint16 d){
    assert(d <= MAT_SIZE);

    depth = d;
    breadth = 1;

    for(uint16 i = 0; i < d; i++){
        data[i][0] = orig_data[i];
    }
}

Matrix::Matrix(float orig_data[MAT_SIZE][MAT_SIZE], uint16 d, uint16 b){
    assert(d <= MAT_SIZE);

    depth = d;
    breadth = b;

    for(uint16 i = 0; i < d; i++){
        for(uint16 j = 0; j < b; j++){
            data[i][j] = orig_data[i][j];
        }
    }
}

void Matrix::print(){
    cout << "[" << endl;

    for(uint16 i = 0; i < depth; i++){
        cout << "[";

        for(uint16 j = 0; j < breadth; j++){
            cout << data[i][j] << ", ";
        }

        cout << "]" << endl;
    }

    cout << "]" << endl;
}

Matrix Matrix::copy(){
    return Matrix(data, depth, breadth);
}

Matrix Matrix::transpose(){
    Matrix ans(breadth, depth);

    for(uint16 i = 0; i < depth; i++){
        for(uint16 j = 0; j < breadth; j++){
            //swap i,j and j,i in the answer
            ans.data[j][i] = data[i][j];
        }
    }

    return ans;
}

Matrix Matrix::add(Matrix b){
    assert(depth == b.depth && breadth == b.breadth);

    Matrix ans(depth, breadth);

    for(uint16 i = 0; i < depth; i++){
        for(uint16 j = 0; j < breadth; j++){
            ans.data[i][j] = data[i][j] + b.data[i][j];
        }
    }

    return ans;
}

Matrix Matrix::sub(Matrix b){
    assert(depth == b.depth &&
            breadth == b.breadth);

    Matrix ans(depth, breadth);

    for(uint16 i = 0; i < depth; i++){
        for(uint16 j = 0; j < breadth; j++){
            ans.data[i][j] = data[i][j] - b.data[i][j];
        }
    }

    return ans;
}

Matrix Matrix::dot(Matrix b){
    assert(breadth == b.depth);
    
    Matrix ans(depth, b.breadth);

    for(uint16 i = 0; i < depth; i++){
        for(uint16 j = 0; j < b.breadth; j++){
            ans.data[i][j] = 0;

            for(uint16 k = 0; k < breadth; k++){
                ans.data[i][j] += data[i][k]*b.data[k][j];
            }
        }
    }

    return ans;
}

Matrix Matrix::mul(Matrix b){
    assert(depth == b.depth &&
            breadth == b.breadth);

    Matrix ans(depth, breadth);

    for(uint16 i = 0; i < depth; i++){
        for(uint16 j = 0; j < breadth; j++){
            ans.data[i][j] = data[i][j] * b.data[i][j];
        }
    }

    return ans;
}



Matrix Matrix::sig(){
    Matrix ans(depth, breadth);

    for(uint16 i = 0; i < depth; i++){
        for(uint16 j = 0; j < breadth; j++){
            ans.data[i][j] = sigmoid(data[i][j]);
        }
    }

    return ans;
}

Matrix Matrix::sigp(){
    Matrix ans(depth, breadth);

    for(uint16 i = 0; i < depth; i++){
        for(uint16 j = 0; j < breadth; j++){
            ans.data[i][j] = sigmoidp(data[i][j]);
        }
    }

    return ans;
}

Matrix Matrix::trunc(){
    Matrix ans(depth, breadth);

    for(uint16 i = 0; i < depth; i++){
        for(uint16 j = 0; j < breadth; j++){
            ans.data[i][j] = truncate(data[i][j]);
        }
    }

    return ans;
}
