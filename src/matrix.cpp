#include "include/matrix.h"

Matrix::Matrix(uint16 d, uint16 b){
    //initialize a 2d matrix of zeros
    depth = d;
    breadth = b;

    for(uint16 i = 0; i < d; i++){
        data.push_back(vector<float>());
        for(uint16 j = 0; j < b; j++){
            data.at(i).push_back(0);
        }
    }
}

void Matrix::patternFill(float a, float b){
    //alternate between a and b in even/odd columns
    for(uint16 i = 0; i < depth; i++){
        for(uint16 j = 0; j < breadth; j++){
            data.at(i).at(j) = (i%2 == 0) a : b;
        }
    }
}

void Matrix::setRow(uint16 row, float* orig_data){
    assert(row < depth);

    //clear the row
    data.at(row) = vector<float>();

    //fill row with given data
    for(uint16 j = 0; j < breadth; j++){
        data.at(row).push_back(orig_data[j]);
    }
}

void Matrix::print(){
    cout << "[" << endl;

    for(uint16 i = 0; i < depth; i++){
        cout << "[";

        for(uint16 j = 0; j < breadth; j++){
            cout << data.at(i).at(j) << ", ";
        }

        cout << "]" << endl;
    }

    cout << "]" << endl;
}

Matrix Matrix::copy(){
    Matrix m(depth, breadth);

    for(uint16 i = 0; i < depth; i++){
        for(uint16 j = 0; j < breadth; j++){
            m.data.at(i).at(j) = data.at(i).at(j);
        }
    }

    return m;
}

Matrix Matrix::transpose(){
    Matrix ans(breadth, depth);

    for(uint16 i = 0; i < depth; i++){
        for(uint16 j = 0; j < breadth; j++){
            //swap i,j and j,i in the answer
            ans.data.at(j).at(i) = data.at(i).at(j);
        }
    }

    return ans;
}

Matrix Matrix::add(Matrix b){
    assert(depth == b.depth && breadth == b.breadth);

    Matrix ans(depth, breadth);

    for(uint16 i = 0; i < depth; i++){
        for(uint16 j = 0; j < breadth; j++){
            ans.data.at(i).at(j) = data.at(i).at(j) + b.data.at(i).at(j);
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
            ans.data.at(i).at(j) = data.at(i).at(j) - b.data.at(i).at(j);
        }
    }

    return ans;
}

Matrix Matrix::dot(Matrix b){
    assert(breadth == b.depth);

    Matrix ans(depth, b.breadth);

    for(uint16 i = 0; i < depth; i++){
        for(uint16 j = 0; j < b.breadth; j++){
            ans.data.at(i).at(j) = 0;

            for(uint16 k = 0; k < breadth; k++){
                ans.data.at(i).at(j) += data.at(i).at(k)*b.data.at(k).at(j);
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
            ans.data.at(i).at(j) = data.at(i).at(j) * b.data.at(i).at(j);
        }
    }

    return ans;
}



Matrix Matrix::sig(){
    Matrix ans(depth, breadth);

    for(uint16 i = 0; i < depth; i++){
        for(uint16 j = 0; j < breadth; j++){
            ans.data.at(i).at(j) = sigmoid(data.at(i).at(j));
        }
    }

    return ans;
}

Matrix Matrix::sigp(){
    Matrix ans(depth, breadth);

    for(uint16 i = 0; i < depth; i++){
        for(uint16 j = 0; j < breadth; j++){
            ans.data.at(i).at(j) = sigmoidp(data.at(i).at(j));
        }
    }

    return ans;
}

Matrix Matrix::trunc(){
    Matrix ans(depth, breadth);

    for(uint16 i = 0; i < depth; i++){
        for(uint16 j = 0; j < breadth; j++){
            ans.data.at(i).at(j) = truncate(data.at(i).at(j));
        }
    }

    return ans;
}
