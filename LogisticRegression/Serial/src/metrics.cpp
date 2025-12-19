#include "metrics.h"
#include <iostream>
constexpr double DECISION_THRESHOLD = 0.50;
using namespace std;


// return Confusion Matrix
ConfusionMatrix computeConfusionMatrix(
    const FeatureMatrix& X,
    const Labels& y,
    const Model& model)
{
    ConfusionMatrix cm{0,0,0,0};

    for (size_t i = 0; i < X.size(); ++i) {
        int pred = (model.predict(X[i]) >= DECISION_THRESHOLD) ? 1 : 0;

        if (pred == 1 && y[i] == 1)      cm.TP++;
        else if (pred == 0 && y[i] == 0) cm.TN++;
        else if (pred == 1 && y[i] == 0) cm.FP++;
        else                             cm.FN++;
    }

    return cm;
}


ModelMetrics computeModelMetrics(const ConfusionMatrix& cm)
{
    ModelMetrics m{};

    double total = cm.TP + cm.TN + cm.FP + cm.FN;

    m.accuracy  = (cm.TP + cm.TN) / total;
    m.precision = (cm.TP + cm.FP) ? double(cm.TP) / (cm.TP + cm.FP)   : 0.0;
    m.recall    = (cm.TP + cm.FN) ? double(cm.TP) / (cm.TP + cm.FN)   : 0.0;
    m.f1        = (m.precision + m.recall)
                     ? 2.0 * (m.precision * m.recall) /
                             (m.precision + m.recall) : 0.0;
    return m;
}