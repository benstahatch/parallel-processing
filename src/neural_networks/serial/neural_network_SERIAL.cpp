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
#include <sys/resource.h>

using namespace std;

// Memory tracking
double getMemoryUsageMB() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss / 1024.0;
}

// Activation functions
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoidDerivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

double relu(double x) {
    return max(0.0, x);
}

double reluDerivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

// Neural Network Class
class NeuralNetwork {
private:
    vector<int> layerSizes;
    vector<vector<vector<double>>> weights;  // weights[layer][neuron][input]
    vector<vector<double>> biases;           // biases[layer][neuron]
    vector<vector<double>> activations;       // activations[layer][neuron]
    vector<vector<double>> zValues;          // z values before activation
    double learningRate;
    mt19937 rng;
    
    // Initialize weights and biases
    void initializeWeights() {
        normal_distribution<double> dist(0.0, 0.1);
        
        for (size_t l = 1; l < layerSizes.size(); ++l) {
            vector<vector<double>> layerWeights;
            vector<double> layerBiases;
            
            for (int i = 0; i < layerSizes[l]; ++i) {
                vector<double> neuronWeights;
                for (int j = 0; j < layerSizes[l - 1]; ++j) {
                    neuronWeights.push_back(dist(rng));
                }
                layerWeights.push_back(neuronWeights);
                layerBiases.push_back(dist(rng));
            }
            
            weights.push_back(layerWeights);
            biases.push_back(layerBiases);
        }
    }
    
    // Forward propagation
    double forward(const vector<double>& input) {
        // Set input layer
        activations.clear();
        zValues.clear();
        activations.push_back(input);
        
        // Forward through hidden layers
        for (size_t l = 0; l < weights.size(); ++l) {
            vector<double> layerActivations;
            vector<double> layerZ;
            
            for (size_t i = 0; i < weights[l].size(); ++i) {
                double z = biases[l][i];
                for (size_t j = 0; j < weights[l][i].size(); ++j) {
                    z += weights[l][i][j] * activations[l][j];
                }
                
                layerZ.push_back(z);
                
                // Apply activation function
                double activation;
                if (l == weights.size() - 1) {  // Output layer
                    activation = sigmoid(z);
                } else {  // Hidden layers
                    activation = relu(z);
                }
                layerActivations.push_back(activation);
            }
            
            zValues.push_back(layerZ);
            activations.push_back(layerActivations);
        }
        
        return activations.back()[0];
    }
    
    // Backward propagation
    void backward(const vector<double>& input, int target, double prediction) {
        // Calculate output layer error
        vector<vector<double>> deltas(weights.size());
        
        // Output layer delta
        double outputError = prediction - target;
        deltas.back().push_back(outputError);
        
        // Backpropagate errors
        for (int l = weights.size() - 2; l >= 0; --l) {
            vector<double> layerDelta;
            for (size_t i = 0; i < weights[l].size(); ++i) {
                double error = 0.0;
                for (size_t j = 0; j < weights[l + 1].size(); ++j) {
                    error += deltas[l + 1][j] * weights[l + 1][j][i];
                }
                layerDelta.push_back(error * reluDerivative(zValues[l][i]));
            }
            deltas[l] = layerDelta;
        }
        
        // Update weights and biases
        for (size_t l = 0; l < weights.size(); ++l) {
            for (size_t i = 0; i < weights[l].size(); ++i) {
                // Update biases
                biases[l][i] -= learningRate * deltas[l][i];
                
                // Update weights
                for (size_t j = 0; j < weights[l][i].size(); ++j) {
                    weights[l][i][j] -= learningRate * deltas[l][i] * activations[l][j];
                }
            }
        }
    }
    
public:
    NeuralNetwork(const vector<int>& layers, double lr = 0.001, int seed = 42)
        : layerSizes(layers), learningRate(lr), rng(seed) {
        initializeWeights();
    }
    
    void train(const vector<vector<double>>& X, const vector<int>& y, 
               int epochs, int logInterval = 10) {
        int m = X.size();
        
        ofstream logFile("serial_cnn_metrics.csv");
        logFile << "epoch,loss,accuracy\n";
        
        cout << "Training Neural Network...\n";
        cout << "Epochs: " << epochs << "\n";
        cout << "Learning rate: " << learningRate << "\n\n";
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double totalLoss = 0.0;
            int correct = 0;
            
            // Train on each sample
            for (int i = 0; i < m; ++i) {
                double prediction = forward(X[i]);
                backward(X[i], y[i], prediction);
                
                // Calculate loss (binary cross-entropy)
                double epsilon = 1e-15;
                prediction = max(min(prediction, 1.0 - epsilon), epsilon);
                totalLoss += -(y[i] * log(prediction) + (1 - y[i]) * log(1 - prediction));
                
                // Calculate accuracy
                int pred = (prediction >= 0.5) ? 1 : 0;
                if (pred == y[i]) correct++;
            }
            
            double avgLoss = totalLoss / m;
            double accuracy = static_cast<double>(correct) / m;
            
            if (epoch % logInterval == 0) {
                logFile << epoch << "," << avgLoss << "," << accuracy << "\n";
                cout << "[Serial CNN] Epoch " << setw(5) << epoch
                     << " | Loss: " << fixed << setprecision(6) << avgLoss
                     << " | Accuracy: " << setprecision(4) << accuracy * 100 << "%\n";
            }
        }
        
        logFile.close();
        cout << "\nTraining completed\n";
    }
    
    vector<int> predict(const vector<vector<double>>& X) {
        vector<int> predictions;
        for (const auto& sample : X) {
            double prob = forward(sample);
            predictions.push_back((prob >= 0.5) ? 1 : 0);
        }
        return predictions;
    }
};

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

// Evaluate Metrics
void evaluateMetrics(const vector<int>& y_true, const vector<int>& y_pred) {
    int TP = 0, TN = 0, FP = 0, FN = 0;
    
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_pred[i] == 1 && y_true[i] == 1) TP++;
        else if (y_pred[i] == 0 && y_true[i] == 0) TN++;
        else if (y_pred[i] == 1 && y_true[i] == 0) FP++;
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

int main() {
    cout << "============================================================\n";
    cout << "Serial Neural Network (CNN)\n";
    cout << "============================================================\n";
    
    vector<vector<double>> X;
    vector<int> y;
    
    // Load dataset
    readCSV("diabetes.csv", X, y);
    cout << "Dataset loaded: " << X.size() << " samples, " << X[0].size() << " features\n";
    
    // Normalize features
    minMaxScale(X);
    
    // Count classes
    int count0 = 0, count1 = 0;
    for (int label : y) {
        if (label == 0) count0++;
        else count1++;
    }
    cout << "Class 0: " << count0 << ", Class 1: " << count1 << "\n\n";
    
    // Define network architecture: 8 -> 64 -> 32 -> 16 -> 1
    int inputSize = X[0].size();
    vector<int> layers = {inputSize, 64, 32, 16, 1};
    
    cout << "Network Architecture: ";
    for (size_t i = 0; i < layers.size(); ++i) {
        cout << layers[i];
        if (i < layers.size() - 1) cout << " -> ";
    }
    cout << "\n\n";
    
    // Train model with timing
    auto start = chrono::high_resolution_clock::now();
    
    NeuralNetwork nn(layers, 0.001, 42);
    nn.train(X, y, 500, 10);
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    
    // Make predictions
    vector<int> predictions = nn.predict(X);
    
    double endMemory = getMemoryUsageMB();
    
    cout << "\n---------------------------------------------------\n";
    cout << "T_serial (seconds): " << elapsed.count() << "\n";
    cout << "Memory Usage (MB): " << endMemory << "\n";
    cout << "---------------------------------------------------\n";
    
    // Evaluate
    evaluateMetrics(y, predictions);
    cout << "---------------------------------------------------\n";
    
    // Save results
    int correct = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        if (predictions[i] == y[i]) correct++;
    }
    double accuracy = static_cast<double>(correct) / y.size();
    
    ofstream results("results_cnn_serial.csv");
    results << "method,time_seconds,memory_mb,accuracy\n";
    results << "Serial CNN," << elapsed.count() << "," << endMemory << "," << accuracy << "\n";
    results.close();
    
    return 0;
}


