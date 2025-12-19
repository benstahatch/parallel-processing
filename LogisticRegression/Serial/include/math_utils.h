#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include "types.h"
#include <cmath>    // library for exp(), log()

// map any real value to (0,1) using Sigmoid function
double sigmoid(double z);

// compute Dot Product (w^T * x)
double dotProduct(const FeatureVector& w, const FeatureVector& x);

// predicts probality for a single same
double predictProbability(const FeatureVector& w, const FeatureVector& x, double b);

// binary cross entropy loss (weighted)
double binaryCrossEntropy(int y, double p, double w0, double w1);


#endif