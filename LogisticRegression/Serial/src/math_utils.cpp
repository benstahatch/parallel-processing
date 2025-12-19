#include "math_utils.h"
#include <cmath>
#include <algorithm>
#include <iostream>


// Sigmoid function
double sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

// compute Dot Product (w^T * x) between two vectors
double dotProduct(const FeatureVector& w, const FeatureVector& x) {
    double sum = 0;
    for (size_t i = 0; i < w.size(); ++i) {
        sum += w[i] * x[i];
    }
    return sum;
    
}

// predicts probality for a single same
double predictProbability(const FeatureVector& w, const FeatureVector& x, double b) {
    return sigmoid(dotProduct(w, x) + b);
}


// binary cross entropy loss (weighted)
double binaryCrossEntropy(int y, double p, double w0, double w1) {
    const double epsilon = 1e-15;
    
    // clamp probability for numerical stability
    double prob = std::min(std::max(p, epsilon), 1.0 - epsilon);
    if (y == 1)
        return -w1 * std::log(prob);
    else
        return -w0 * std::log(1.0 - prob);
}

    







