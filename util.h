#pragma once

#ifndef NEUNET_H
#define NEUNET_H

#include "stdio.h"
#include "stdlib.h"
#include "cassert.h"
#include "cmath.h"
#include "random.h"

using namespace std;

typedef unsigned short int uint16;
typedef unsigned int uint32;

typedef unsigned char neum; //represents a decimal between 0 and 1

const uint16 NEUM_LIM = 255;

float NtoF(neum v);
neum FtoN(float v);

neum sig(float z);
float sigp(float z);

float cost(float a, float y);

void copy(neum* a, neum* b, uint16 size);

void transpose(neum* arr, uint16 size);

void mul(float fac, neum* arr, uint16 size);

#endif
