#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "model.h"

class LogisticRegression : public Model {
private:
    FeatureVector weights;  // Model weights
    double bias;            // bias term
    double learningRate;
    int epochs;


public:

    // constructor
    LogisticRegression(int numFeatures, double alpha = 0.01, int epochs = 5000);


    // override base class functions
    void train(const FeatureMatrix& X, const Labels& y) override;


    double predict(const FeatureVector& x) const override;

    // accessors (for metrics)
    const FeatureVector& getWeights() const;
    double getBias() const;

};


#endif