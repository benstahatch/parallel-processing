#include "output.h"
#include <iostream>
#include <iomanip>
using namespace std;




void printMetricsReport(const ConfusionMatrix& cm,const ModelMetrics& metrics, const PerformanceMetrics& perf)
{

    cout << "\n---- Metrics ----\n";
    cout << "TP: " << cm.TP << "   TN: " << cm.TN << endl;
    cout << "FP: " << cm.FP << "   FN: " << cm.FN << endl;
    cout << fixed << setprecision(4);
    cout << "Accuracy:   " << metrics.accuracy * 100 << "%" << endl;

    cout << "\n===== FULL MODEL METRICS =====\n";
    cout << "Accuracy:   " << metrics.accuracy * 100 << "%" << endl;
    cout << "Precision:  " << metrics.precision * 100 << "%" << endl;
    cout << "Recall:     " << metrics.recall * 100 << "%" << endl;
    cout << "Overhead:   " << perf.overheadPercent << "%" << endl;
    cout << "F1 Score:   " << metrics.f1 * 100 << "%" << endl;
    cout << "===================================" << endl << endl;

    cout << "\n----------------------------------------------------\n";
    cout << "T_serial (seconds): " << perf.timeSeconds << endl;
    cout << "Memory Usage (MB):  " << perf.memoryMB << endl;
    cout << "MPI Ranks: "          << perf.mpiRanks << endl;
    cout << "----------------------------------------------------\n";




}