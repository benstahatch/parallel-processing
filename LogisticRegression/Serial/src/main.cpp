#include "dataset.h"
#include "logistic_regression.h"
#include "metrics.h"
#include "performace.h"
#include "output.h"
#include <chrono>
#include <iostream>
using namespace std;




int main() {
    FeatureMatrix X;
    Labels y;


    cout << "============================================================\n";
    cout << "Serial Logistic Regression\n";
    cout << "============================================================\n";

    // load and preprocess dataset
    loadCSV("diabetes.csv", X, y);
    minMaxScale(X);

    cout << "Loaded dataset: " << X.size()
         << " samples, " << X[0].size()
         << " features\n";
    cout << "Training logistic regression using serial execution...\n\n";


    // model polymorphism
    Model* model = new LogisticRegression(X[0].size());

    // time training
    auto start = chrono::high_resolution_clock::now();
    model->train(X, y);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;



    // evaluate model
    ConfusionMatrix cm = computeConfusionMatrix(X, y, *model);
    ModelMetrics metrics = computeModelMetrics(cm);


    // performace metrics (serial baseline)
    PerformanceMetrics perf;
    perf.timeSeconds = elapsed.count();
    perf.memoryMB = getMemoryUsageMB();
    perf.mpiRanks = 1;          // serial baseline
    perf.overheadPercent = 0.0; // serial baseline



    // final formatted output
    printMetricsReport(cm, metrics, perf);

    delete model;


    return 0;
}
