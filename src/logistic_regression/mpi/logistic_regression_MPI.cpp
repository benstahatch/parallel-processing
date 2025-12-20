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
#include <mpi.h>

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

// Gradient Descent Epoch - computes gradients for local data
void gradientDescentEpoch(const vector<vector<double>>& X_local, const vector<int>& y_local,
                         const vector<double>& w, double b, double alpha, double w0, double w1,
                         vector<double>& dw_local, double& db_local, int m_total) {
    const int m_local = static_cast<int>(X_local.size());
    const int n = static_cast<int>(w.size());
    
    fill(dw_local.begin(), dw_local.end(), 0.0);
    db_local = 0.0;
    
    for (int i = 0; i < m_local; ++i) {
        double p = predictProbability(w, X_local[i], b);
        double error = static_cast<double>(y_local[i]) - p;
        double sampleWeight = (y_local[i] == 1) ? w1 : w0;
        
        for (int j = 0; j < n; ++j) {
            dw_local[j] += sampleWeight * error * X_local[i][j];
        }
        db_local += sampleWeight * error;
    }
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
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        cout << "============================================================\n";
        cout << "MPI Logistic Regression\n";
        cout << "============================================================\n";
        cout << "MPI Processes: " << size << "\n";
    }
    
    vector<vector<double>> X_full, X_local;
    vector<int> y_full, y_local;
    int m_total, n_features;
    double alpha = 0.01;
    int epochs = 5000;
    
    // Rank 0 loads data
    if (rank == 0) {
        readCSV("diabetes.csv", X_full, y_full);
        minMaxScale(X_full);
        m_total = X_full.size();
        n_features = X_full[0].size();
        
        cout << "Dataset loaded: " << m_total << " samples, " << n_features << " features\n";
        
        int count0 = 0, count1 = 0;
        for (int label : y_full) {
            if (label == 0) count0++;
            else count1++;
        }
        cout << "Class 0: " << count0 << ", Class 1: " << count1 << "\n\n";
    }
    
    // Broadcast dimensions
    MPI_Bcast(&m_total, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_features, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Distribute data among processes
    int m_local = m_total / size;
    int remainder = m_total % size;
    if (rank < remainder) m_local++;
    
    X_local.resize(m_local, vector<double>(n_features));
    y_local.resize(m_local);
    
    // Scatter data
    if (rank == 0) {
        int offset = 0;
        for (int r = 0; r < size; ++r) {
            int send_count = m_total / size;
            if (r < remainder) send_count++;
            
            if (r == 0) {
                for (int i = 0; i < send_count; ++i) {
                    X_local[i] = X_full[i];
                    y_local[i] = y_full[i];
                }
            } else {
                for (int i = 0; i < send_count; ++i) {
                    for (int j = 0; j < n_features; ++j) {
                        MPI_Send(&X_full[offset + i][j], 1, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
                    }
                    MPI_Send(&y_full[offset + i], 1, MPI_INT, r, 1, MPI_COMM_WORLD);
                }
            }
            offset += send_count;
        }
    } else {
        for (int i = 0; i < m_local; ++i) {
            for (int j = 0; j < n_features; ++j) {
                MPI_Recv(&X_local[i][j], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            MPI_Recv(&y_local[i], 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    
    // Calculate class weights (on rank 0)
    double w0, w1;
    if (rank == 0) {
        int count1 = count(y_full.begin(), y_full.end(), 1);
        int count0 = m_total - count1;
        w1 = static_cast<double>(m_total) / (2.0 * max(1, count1));
        w0 = static_cast<double>(m_total) / (2.0 * max(1, count0));
    }
    MPI_Bcast(&w0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&w1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Initialize weights
    vector<double> w(n_features, 0.0);
    double b = 0.0;
    
    ofstream logFile;
    if (rank == 0) {
        logFile.open("mpi_lr_metrics.csv");
        logFile << "epoch,loss,accuracy\n";
    }
    
    auto start = chrono::high_resolution_clock::now();
    
    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Compute local gradients
        vector<double> dw_local(n_features);
        double db_local;
        gradientDescentEpoch(X_local, y_local, w, b, alpha, w0, w1, dw_local, db_local, m_total);
        
        // Reduce gradients
        vector<double> dw_global(n_features);
        double db_global;
        MPI_Allreduce(dw_local.data(), dw_global.data(), n_features, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&db_local, &db_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        // Update parameters
        const double inv_m = 1.0 / static_cast<double>(m_total);
        for (int j = 0; j < n_features; ++j) {
            w[j] += alpha * dw_global[j] * inv_m;
        }
        b += alpha * db_global * inv_m;
        
        // Log metrics every 100 epochs
        if (epoch % 100 == 0) {
            double local_loss = 0.0;
            int local_correct = 0;
            
            for (int i = 0; i < m_local; ++i) {
                double p = predictProbability(w, X_local[i], b);
                local_loss += binaryCrossEntropy(y_local[i], p, w0, w1);
                int pred = (p >= 0.5) ? 1 : 0;
                if (pred == y_local[i]) local_correct++;
            }
            
            double global_loss, global_correct;
            MPI_Reduce(&local_loss, &global_loss, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            double temp_correct = static_cast<double>(local_correct);
            MPI_Reduce(&temp_correct, &global_correct, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            
            if (rank == 0) {
                double avgLoss = global_loss / m_total;
                double accuracy = global_correct / m_total;
                
                logFile << epoch << "," << avgLoss << "," << accuracy << "\n";
                cout << "[MPI LR] Epoch " << setw(5) << epoch
                     << " | Loss: " << fixed << setprecision(6) << avgLoss
                     << " | Accuracy: " << setprecision(4) << accuracy * 100 << "%\n";
            }
        }
    }
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    
    if (rank == 0) {
        logFile.close();
        
        double endMemory = getMemoryUsageMB();
        
        cout << "\n---------------------------------------------------\n";
        cout << "T_mpi (seconds): " << elapsed.count() << "\n";
        cout << "Memory Usage (MB): " << endMemory << "\n";
        cout << "MPI Processes: " << size << "\n";
        cout << "---------------------------------------------------\n";
        
        // Final evaluation on full dataset
        int TP = 0, TN = 0, FP = 0, FN = 0;
        for (size_t i = 0; i < X_full.size(); ++i) {
            double p = predictProbability(w, X_full[i], b);
            int pred = (p >= 0.5) ? 1 : 0;
            if (pred == 1 && y_full[i] == 1) TP++;
            else if (pred == 0 && y_full[i] == 0) TN++;
            else if (pred == 1 && y_full[i] == 0) FP++;
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
        cout << "---------------------------------------------------\n";
        
        // Save results
        ofstream results("results_lr_mpi.csv");
        results << "method,time_seconds,memory_mb,processes,accuracy\n";
        results << "MPI LR," << elapsed.count() << "," << endMemory << "," << size << "," << accuracy << "\n";
        results.close();
    }
    
    MPI_Finalize();
    return 0;
}

