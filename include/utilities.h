#pragma once
#include <cstdlib>
#include <iostream>
#include <vector>
#include <array>
#include <opencv2/imgproc.hpp>

void fitQuadratic(float x1, float y1, float x2, float y2, float x3, float y3, float &a, float &b, float &c);

void threshToZero(cv::Mat &inImg, const float threshold);

void convertToGray(const cv::Mat &inImg, cv::Mat &outImg);

void applyGaussianBlur(const cv::Mat &inImg, cv::Mat &outImg, const unsigned int kernel_size);

void sumOverKernel(cv::Mat &inImg, cv::Mat &outImg, const unsigned int kernel_size);

void computeModifiedLaplace(const cv::Mat &inImg, cv::Mat &lap);

