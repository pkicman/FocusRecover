#pragma once
#include <cstdlib>
#include <iostream>
#include <vector>
#include <array>
#include "myMat.h"

void fitQuadratic(float x1, float y1, float x2, float y2, float x3, float y3, float &a, float &b, float &c);

void threshToZero(myMat<float> &inImg, const float threshold);

void convertToGray(const myMat<uchar> &inImg, myMat<uchar> &outImg);

void applyGaussianBlur(const myMat<uchar> &inImg, myMat<uchar> &outImg, const unsigned int kernel_size);

void sumOverKernel(myMat<float> &inImg, myMat<float> &outImg, const unsigned int kernel_size);

void computeModifiedLaplace(const myMat<uchar> &inImg, myMat<float> &lap);

