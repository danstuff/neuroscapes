#pragma once

#ifndef UTIL_H
#define UTIL_H

#include "stdio.h"
#include "stdlib.h"
#include "assert.h"
#include "math.h"
#include <iostream>

using namespace std;

typedef unsigned short int uint16;
typedef unsigned int uint32;

typedef unsigned char neum; //represents a decimal between 0 and 1

const uint16 NEUM_LIM = 255;

const uint16 RAND_MUL = 1000;
const uint16 RAND_SEED = 123;

float NtoF(neum v);
neum FtoN(float v);

float randf(float min, float max);

neum sig(float z);
float sigp(float z);

void copy(float* a, float* b, uint16 size);
void copy2d(float** a, float** b, uint16 size);

void transpose(float** arr, float** ans, uint16 d, uint16 b);

void dot(float** a, float** b, float** ans, uint16 da, uint16 ba, uint16 db, uint16 bb);

#endif
