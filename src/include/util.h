#pragma once

#ifndef UTIL_H
#define UTIL_H

#include "stdio.h"
#include "stdlib.h"
#include "assert.h"
#include "math.h"
#include <iostream>
#include <fstream>

using namespace std;

typedef unsigned short int uint16;
typedef unsigned int uint32;

const uint16 RAND_MUL = 1000;
const uint16 RAND_SEED = 123;

const uint16 ROUND_MUL = 1000;

float randf(float min, float max);

float sigmoid(float z);
float sigmoidp(float z);

float truncate(float v);

#endif
