#ifndef OUTPUT_H
#define OUTPUT_H

#include "metrics.h"
#include "performace.h"

// print out the final model's metrics

void printMetricsReport(const ConfusionMatrix& cm,const ModelMetrics& metrics, const PerformanceMetrics& perf);


#endif