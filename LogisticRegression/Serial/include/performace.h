#ifndef PERFORMANCE_H
#define PERFORMANCE_H

// system level perfomance information
struct PerformanceMetrics {
    double timeSeconds;         // wall clock execution timing
    double memoryMB;            // memory usage in MB
    int mpiRanks;               // serial has 1 rank
    double overheadPercent;     // serial (0.0)
};

// return existing memory usage of the process in MB
double getMemoryUsageMB();



#endif