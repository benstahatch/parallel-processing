#ifndef MODEL_H
#define MODEL_H

#include "types.h"


// abstract base class for all ML models
class Model {
public:
    virtual ~Model() = default;

    // train the model using the data set
    virtual void train(const FeatureMatrix& X, const Labels& y) = 0;


    // predict probability for a single sample
    virtual double predict(const FeatureVector& x) const = 0;
};


#endif