#include <iostream>        
#include <iomanip>
#include <vector>           
#include <string>           
#include <fstream>         
#include <sstream>         
#include <cmath>          
#include <algorithm>        
#include <chrono>          
#include <random>
#include <numeric>
#include <sys/resource.h>
#include <omp.h>

using namespace std;

// Memory tracking function
double getMemoryUsageMB() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss / 1024.0;
}

// Sigmoid Function
double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

// Dot Product
inline double dotProduct(const vector<double>& w, const vector<double>& x) {
    double sumation = 0.0;
    #pragma omp parallel for reduction(+:sumation)
    for (size_t j = 0; j < w.size(); ++j) {
        sumation += w[j] * x[j];
    }
    return sumation;
}

// Predict Probability
inline double predictProbability(const vector<double>& w, const vector<double>& x, double b) {
    double z = dotProduct(w, x) + b;
    return sigmoid(z);
}

// Binary Cross Entropy
inline double binaryCrossEntropy(int y, double p, double w0, double w1) {
    const double epsilon = 1e-15;
    double probability = min(max(p, epsilon), 1.0 - epsilon);
    return (y == 1) ? -w1 * log(probability) : -w0 * log(1.0 - probability);
}

// Gradient Descent Epoch with OpenMP parallelization
void gradientDescentEpoch(const vector<vector<double>>& X, const vector<int>& y,
                         vector<double>& w, double& b, double alpha, double w0, double w1) {
    const int m = static_cast<int>(X.size());
    const int n = static_cast<int>(w.size());
    
    vector<double> dw(n, 0.0);
    double db = 0.0;
    
    // Parallelize gradient accumulation
    #pragma omp parallel
    {
        vector<double> dw_local(n, 0.0);
        double db_local = 0.0;
        
        #pragma omp for nowait
        for (int i = 0; i < m; ++i) {
            double p = predictProbability(w, X[i], b);
            double error = static_cast<double>(y[i]) - p;
            double sampleWeight = (y[i] == 1) ? w1 : w0;
            
            for (int j = 0; j < n; ++j) {
                dw_local[j] += sampleWeight * error * X[i][j];
            }
            db_local += sampleWeight * error;
        }
        
        // Reduce local gradients to global
        #pragma omp critical
        {
            for (int j = 0; j < n; ++j) {
                dw[j] += dw_local[j];
            }
            db += db_local;
        }
    }
    
    // Update parameters
    const double inv_m = 1.0 / static_cast<double>(m);
    #pragma omp parallel for
    for (int j = 0; j < n; ++j) {
        w[j] += alpha * dw[j] * inv_m;
    }
    b += alpha * db * inv_m;
}

// Train Model
void trainModel_LogCSV(vector<vector<double>>& X, vector<int>& y, vector<double>& w, 
                      double& b, double alpha, int epochs, int num_threads) {
    const int m = static_cast<int>(X.size());
    
    const int count1 = static_cast<int>(count(y.begin(), y.end(), 1));
    const int count0 = m - count1;
    const double w1 = static_cast<double>(m) / (2.0 * max(1, count1));
    const double w0 = static_cast<double>(m) / (2.0 * max(1, count0));
    
    ofstream logFile("openmp_lr_metrics.csv");
    logFile << "epoch,loss,accuracy\n";
    
    cout << "Using " << num_threads << " OpenMP threads\n\n";
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        gradientDescentEpoch(X, y, w, b, alpha, w0, w1);
        
        if (epoch % 100 == 0) {
            double totalLoss = 0.0;
            int correct = 0;
            
            // Parallelize loss and accuracy computation
            #pragma omp parallel for reduction(+:totalLoss,correct)
            for (int i = 0; i < m; ++i) {
                double p = predictProbability(w, X[i], b);
                totalLoss += binaryCrossEntropy(y[i], p, w0, w1);
                int pred = (p >= 0.5) ? 1 : 0;
                if (pred == y[i]) correct++;
            }
            
            double avgLoss = totalLoss / m;
            double accuracy = static_cast<double>(correct) / m;
            
            logFile << epoch << "," << avgLoss << "," << accuracy << "\n";
            cout << "[OpenMP LR] Epoch " << setw(5) << epoch
                 << " | Loss: " << fixed << setprecision(6) << avgLoss
                 << " | Accuracy: " << setprecision(4) << accuracy * 100 << "%\n";
        }
    }
    logFile.close();
}

// Evaluate Metrics
void evaluateMetrics(const vector<vector<double>>& X, const vector<int>& y, 
                    const vector<double>& w, double b) {
    int TP = 0, TN = 0, FP = 0, FN = 0;
    
    #pragma omp parallel for reduction(+:TP,TN,FP,FN)
    for (size_t i = 0; i < X.size(); ++i) {
        double p = predictProbability(w, X[i], b);
        int pred = (p >= 0.5) ? 1 : 0;
        if (pred == 1 && y[i] == 1) TP++;
        else if (pred == 0 && y[i] == 0) TN++;
        else if (pred == 1 && y[i] == 0) FP++;
        else FN++;
    }
    
    double accuracy = static_cast<double>(TP + TN) / (TP + TN + FP + FN);
    double precision = (TP + FP) ? static_cast<double>(TP) / (TP + FP) : 0.0;
    double recall = (TP + FN) ? static_cast<double>(TP) / (TP + FN) : 0.0;
    double f1 = (precision + recall) ? 2.0 * (precision * recall) / (precision + recall) : 0.0;
    
    cout << "\n---- Confusion Matrix Metrics ----\n";
    cout << "---------------------------------------------------\n";
    cout << "Accuracy:  " << accuracy * 100 << "%\n";
    cout << "Precision: " << precision * 100 << "%\n";
    cout << "Recall:    " << recall * 100 << "%\n";
    cout << "F1 Score:  " << f1 * 100 << "%\n";
}

// Read CSV
void readCSV(const string& filename, vector<vector<double>>& X, vector<int>& y) {
    ifstream file(filename);
    string line;
    bool skipHeader = true;
    
    while (getline(file, line)) {
        if (skipHeader) {
            skipHeader = false;
            continue;
        }
        stringstream ss(line);
        string value;
        vector<double> row;
        while (getline(ss, value, ',')) {
            row.push_back(stod(value));
        }
        y.push_back(static_cast<int>(row.back()));
        row.pop_back();
        X.push_back(row);
    }
    file.close();
}

// Min-Max Scaling
void minMaxScale(vector<vector<double>>& X) {
    const int n_features = static_cast<int>(X[0].size());
    for (int j = 0; j < n_features; ++j) {
        double minVal = X[0][j], maxVal = X[0][j];
        for (size_t i = 1; i < X.size(); ++i) {
            minVal = min(minVal, X[i][j]);
            maxVal = max(maxVal, X[i][j]);
        }
        for (size_t i = 0; i < X.size(); ++i) {
            X[i][j] = (maxVal != minVal) ? ((X[i][j] - minVal) / (maxVal - minVal)) : 0.0;
        }
    }
}

int main(int argc, char* argv[]) {
    // Set number of threads (default: all available)
    int num_threads = omp_get_max_threads();
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }
    omp_set_num_threads(num_threads);
    
    cout << "============================================================\n";
    cout << "OpenMP Logistic Regression\n";
    cout << "============================================================\n";
    
    vector<vector<double>> X;
    vector<int> y;
    
    // Load dataset
    readCSV("diabetes.csv", X, y);
    cout << "Dataset loaded: " << X.size() << " samples, " << X[0].size() << " features\n";
    
    // Normalize features
    minMaxScale(X);
    
    // Set parameters
    int n_features = X[0].size();
    vector<double> w(n_features, 0.0);
    double b = 0.0;
    double alpha = 0.01;
    int epochs = 5000;
    
    // Count classes
    int count0 = 0, count1 = 0;
    for (int label : y) {
        if (label == 0) count0++;
        else count1++;
    }
    cout << "Class 0: " << count0 << ", Class 1: " << count1 << "\n\n";
    
    // Train model with timing
    auto start = chrono::high_resolution_clock::now();
    trainModel_LogCSV(X, y, w, b, alpha, epochs, num_threads);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    
    double endMemory = getMemoryUsageMB();
    
    cout << "\n---------------------------------------------------\n";
    cout << "T_openmp (seconds): " << elapsed.count() << "\n";
    cout << "Memory Usage (MB): " << endMemory << "\n";
    cout << "Threads used: " << num_threads << "\n";
    cout << "---------------------------------------------------\n";
    
    // Evaluate
    evaluateMetrics(X, y, w, b);
    cout << "---------------------------------------------------\n";
    
    // Save results
    ofstream results("results_lr_openmp.csv");
    results << "method,time_seconds,memory_mb,threads,accuracy\n";
    results << "OpenMP LR," << elapsed.count() << "," << endMemory << "," << num_threads << ",";
    
    // Calculate accuracy for results file
    int correct = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        double p = predictProbability(w, X[i], b);
        int pred = (p >= 0.5) ? 1 : 0;
        if (pred == y[i]) correct++;
    }
    results << static_cast<double>(correct) / X.size() << "\n";
    results.close();
    
    return 0;
}

