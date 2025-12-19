#include "logistic_regression.h"
#include "math_utils.h"
#include <algorithm>
#include <iostream>
#include <iomanip>



LogisticRegression::LogisticRegression(int numFeatures, double alpha, int epochs)
     : weights(numFeatures, 0.0), bias(0.0), learningRate(alpha), epochs(epochs) {
     }

void LogisticRegression::train(const FeatureMatrix& X, const Labels& y)
{

    const int m = static_cast<int>(X.size());
    const int n = static_cast<int>(weights.size());

    int count1 = std::count(y.begin(), y.end(), 1);
    int count0 = m - count1;

    // compute class weights to handle imbalance
    double w1 = static_cast<double>(m) / (2.0 * std::max(1, count1));
    double w0 = static_cast<double>(m) / (2.0 * std::max(1, count0));

    for (int epoch = 0; epoch < epochs; ++epoch) {
        FeatureVector dw(n, 0.0);
        double db = 0.0;

        for (int i = 0; i < m; ++i) {
            double p = predictProbability(weights, X[i], bias);
            double error = static_cast<double>(y[i]) - p;
            double sampleWeight = (y[i] == 1) ? w1 : w0;

        for (int j = 0; j < n; j++)
            dw[j] += sampleWeight * error * X[i][j];

            db += sampleWeight * error;
        }

        // gradient update
        for (int j = 0; j < n; ++j)
            weights[j] += learningRate * dw[j] / m;
        bias += learningRate * db / m;

        // epoch logging
        if (epoch % 100 == 0) {
            double totalLoss = 0.0;
            int correct = 0;

            for (int i = 0; i < m; ++i) {
                double p = predictProbability(weights, X[i], bias);
                totalLoss += binaryCrossEntropy(y[i], p, w0, w1);
                
                int pred = (p >= 0.5) ? 1 : 0;
                if (pred == y[i]) correct++;
            }

            double avgLoss = totalLoss / m;
            double accuracy = static_cast<double>(correct) / m;

            std::cout << std::fixed
                << "[Serial LR] Epoch: " << std::setw(5) << epoch
                << " | Loss: " << std::setprecision(6) << avgLoss
                << " | Accuracy: " << std::setprecision(4)
                << accuracy * 100 << "%\n";
    
}

    
    }
}





double LogisticRegression::predict(const FeatureVector& x) const {
    return predictProbability(weights, x , bias);
}


const FeatureVector& LogisticRegression::getWeights() const {
    return weights;
}
    double LogisticRegression::getBias() const {
        return bias;
    }