#ifndef METRICS_H
#define METRICS_H

#include "types.h"
#include "model.h"


// True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN)
struct ConfusionMatrix {
    int TP;
    int TN;
    int FP;
    int FN;
};


struct ModelMetrics {
    double accuracy;
    double precision;
    double recall;
    double f1;
};


ConfusionMatrix computeConfusionMatrix(
    const FeatureMatrix& X,
    const Labels& y,
    const Model& model
);


ModelMetrics computeModelMetrics(const ConfusionMatrix& cm);




#endif