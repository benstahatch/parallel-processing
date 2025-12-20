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
#include <map>
#include <set>
#include <sys/resource.h>
#include <omp.h>

using namespace std;

// Memory tracking
double getMemoryUsageMB() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss / 1024.0;
}

// Decision Tree Node
struct TreeNode {
    bool isLeaf;
    int predictedClass;
    int featureIndex;
    double threshold;
    TreeNode* left;
    TreeNode* right;
    
    TreeNode() : isLeaf(false), predictedClass(0), featureIndex(-1), 
                 threshold(0.0), left(nullptr), right(nullptr) {}
    
    ~TreeNode() {
        delete left;
        delete right;
    }
};

// Decision Tree Class
class DecisionTree {
private:
    int maxDepth;
    int minSamplesSplit;
    TreeNode* root;
    mt19937 rng;
    
    double giniImpurity(const vector<int>& labels) {
        if (labels.empty()) return 0.0;
        
        map<int, int> counts;
        for (int label : labels) {
            counts[label]++;
        }
        
        double impurity = 1.0;
        for (auto& pair : counts) {
            double prob = static_cast<double>(pair.second) / labels.size();
            impurity -= prob * prob;
        }
        return impurity;
    }
    
    int mostCommonClass(const vector<int>& labels) {
        map<int, int> counts;
        for (int label : labels) {
            counts[label]++;
        }
        
        int maxCount = 0, commonClass = 0;
        for (auto& pair : counts) {
            if (pair.second > maxCount) {
                maxCount = pair.second;
                commonClass = pair.first;
            }
        }
        return commonClass;
    }
    
    void splitData(const vector<vector<double>>& X, const vector<int>& y,
                   int featureIdx, double threshold,
                   vector<vector<double>>& X_left, vector<int>& y_left,
                   vector<vector<double>>& X_right, vector<int>& y_right) {
        for (size_t i = 0; i < X.size(); ++i) {
            if (X[i][featureIdx] <= threshold) {
                X_left.push_back(X[i]);
                y_left.push_back(y[i]);
            } else {
                X_right.push_back(X[i]);
                y_right.push_back(y[i]);
            }
        }
    }
    
    bool findBestSplit(const vector<vector<double>>& X, const vector<int>& y,
                       const vector<int>& featureIndices,
                       int& bestFeature, double& bestThreshold) {
        double bestGain = -1.0;
        double currentImpurity = giniImpurity(y);
        int m = X.size();
        
        for (int featureIdx : featureIndices) {
            set<double> uniqueVals;
            for (const auto& sample : X) {
                uniqueVals.insert(sample[featureIdx]);
            }
            
            for (double threshold : uniqueVals) {
                vector<vector<double>> X_left, X_right;
                vector<int> y_left, y_right;
                splitData(X, y, featureIdx, threshold, X_left, y_left, X_right, y_right);
                
                if (y_left.empty() || y_right.empty()) continue;
                
                double n_left = y_left.size();
                double n_right = y_right.size();
                double weightedImpurity = (n_left / m) * giniImpurity(y_left) +
                                         (n_right / m) * giniImpurity(y_right);
                double gain = currentImpurity - weightedImpurity;
                
                if (gain > bestGain) {
                    bestGain = gain;
                    bestFeature = featureIdx;
                    bestThreshold = threshold;
                }
            }
        }
        
        return bestGain > 0;
    }
    
    TreeNode* buildTree(const vector<vector<double>>& X, const vector<int>& y,
                       int depth, const vector<int>& featureIndices) {
        TreeNode* node = new TreeNode();
        
        if (depth >= maxDepth || y.size() < static_cast<size_t>(minSamplesSplit) ||
            giniImpurity(y) == 0.0) {
            node->isLeaf = true;
            node->predictedClass = mostCommonClass(y);
            return node;
        }
        
        int bestFeature;
        double bestThreshold;
        if (!findBestSplit(X, y, featureIndices, bestFeature, bestThreshold)) {
            node->isLeaf = true;
            node->predictedClass = mostCommonClass(y);
            return node;
        }
        
        vector<vector<double>> X_left, X_right;
        vector<int> y_left, y_right;
        splitData(X, y, bestFeature, bestThreshold, X_left, y_left, X_right, y_right);
        
        node->isLeaf = false;
        node->featureIndex = bestFeature;
        node->threshold = bestThreshold;
        node->left = buildTree(X_left, y_left, depth + 1, featureIndices);
        node->right = buildTree(X_right, y_right, depth + 1, featureIndices);
        
        return node;
    }
    
    int predictSample(const vector<double>& x, TreeNode* node) {
        if (node->isLeaf) {
            return node->predictedClass;
        }
        
        if (x[node->featureIndex] <= node->threshold) {
            return predictSample(x, node->left);
        } else {
            return predictSample(x, node->right);
        }
    }
    
public:
    DecisionTree(int maxDepth = 10, int minSamplesSplit = 2, int seed = 42)
        : maxDepth(maxDepth), minSamplesSplit(minSamplesSplit), root(nullptr), rng(seed) {}
    
    ~DecisionTree() {
        delete root;
    }
    
    void fit(const vector<vector<double>>& X, const vector<int>& y, 
             const vector<int>& featureIndices) {
        root = buildTree(X, y, 0, featureIndices);
    }
    
    vector<int> predict(const vector<vector<double>>& X) {
        vector<int> predictions;
        for (const auto& sample : X) {
            predictions.push_back(predictSample(sample, root));
        }
        return predictions;
    }
};

// Random Forest Class
class RandomForest {
private:
    int nEstimators;
    int maxDepth;
    int minSamplesSplit;
    int maxFeatures;
    bool bootstrap;
    vector<DecisionTree*> trees;
    mt19937 rng;
    
    void getBootstrapSample(const vector<vector<double>>& X, const vector<int>& y,
                           vector<vector<double>>& X_sample, vector<int>& y_sample, int seed) {
        mt19937 local_rng(seed);
        uniform_int_distribution<int> dist(0, X.size() - 1);
        for (size_t i = 0; i < X.size(); ++i) {
            int idx = dist(local_rng);
            X_sample.push_back(X[idx]);
            y_sample.push_back(y[idx]);
        }
    }
    
    vector<int> getRandomFeatures(int nFeatures, int seed) {
        vector<int> allFeatures;
        for (int i = 0; i < nFeatures; ++i) {
            allFeatures.push_back(i);
        }
        
        mt19937 local_rng(seed);
        shuffle(allFeatures.begin(), allFeatures.end(), local_rng);
        
        int n_select = min(maxFeatures, nFeatures);
        return vector<int>(allFeatures.begin(), allFeatures.begin() + n_select);
    }
    
public:
    RandomForest(int nEstimators = 100, int maxDepth = 10, int minSamplesSplit = 2,
                int maxFeatures = -1, bool bootstrap = true, int seed = 42)
        : nEstimators(nEstimators), maxDepth(maxDepth), minSamplesSplit(minSamplesSplit),
          maxFeatures(maxFeatures), bootstrap(bootstrap), rng(seed) {
        trees.resize(nEstimators, nullptr);
    }
    
    ~RandomForest() {
        for (auto tree : trees) {
            delete tree;
        }
    }
    
    void fit(const vector<vector<double>>& X, const vector<int>& y) {
        int nFeatures = X[0].size();
        if (maxFeatures == -1) {
            maxFeatures = static_cast<int>(sqrt(nFeatures));
        }
        
        cout << "Training " << nEstimators << " trees with OpenMP...\n";
        cout << "Max depth: " << maxDepth << "\n";
        cout << "Max features: " << maxFeatures << "\n";
        cout << "OpenMP threads: " << omp_get_max_threads() << "\n\n";
        
        // Parallel tree training
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < nEstimators; ++i) {
            // Bootstrap sampling
            vector<vector<double>> X_sample;
            vector<int> y_sample;
            if (bootstrap) {
                getBootstrapSample(X, y, X_sample, y_sample, 42 + i);
            } else {
                X_sample = X;
                y_sample = y;
            }
            
            // Random feature selection
            vector<int> featureIndices = getRandomFeatures(nFeatures, 42 + i);
            
            // Train tree
            DecisionTree* tree = new DecisionTree(maxDepth, minSamplesSplit, 42 + i);
            tree->fit(X_sample, y_sample, featureIndices);
            trees[i] = tree;
            
            #pragma omp critical
            {
                if ((i + 1) % 10 == 0) {
                    cout << "Trained " << (i + 1) << "/" << nEstimators << " trees\n";
                }
            }
        }
        cout << "\nCompleted training " << nEstimators << " trees\n";
    }
    
    vector<int> predict(const vector<vector<double>>& X) {
        vector<int> finalPredictions(X.size(), 0);
        
        // Get predictions from all trees in parallel
        vector<vector<int>> allPredictions(nEstimators);
        
        #pragma omp parallel for
        for (int t = 0; t < nEstimators; ++t) {
            allPredictions[t] = trees[t]->predict(X);
        }
        
        // Majority voting
        #pragma omp parallel for
        for (size_t i = 0; i < X.size(); ++i) {
            map<int, int> votes;
            for (int t = 0; t < nEstimators; ++t) {
                votes[allPredictions[t][i]]++;
            }
            
            int maxVotes = 0, prediction = 0;
            for (auto& pair : votes) {
                if (pair.second > maxVotes) {
                    maxVotes = pair.second;
                    prediction = pair.first;
                }
            }
            finalPredictions[i] = prediction;
        }
        
        return finalPredictions;
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
    
    #pragma omp parallel for reduction(+:TP,TN,FP,FN)
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

int main(int argc, char* argv[]) {
    // Set number of threads
    int num_threads = omp_get_max_threads();
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }
    omp_set_num_threads(num_threads);
    
    cout << "============================================================\n";
    cout << "OpenMP Random Forest\n";
    cout << "============================================================\n";
    
    vector<vector<double>> X;
    vector<int> y;
    
    readCSV("diabetes.csv", X, y);
    cout << "Dataset loaded: " << X.size() << " samples, " << X[0].size() << " features\n";
    
    minMaxScale(X);
    
    int count0 = 0, count1 = 0;
    for (int label : y) {
        if (label == 0) count0++;
        else count1++;
    }
    cout << "Class 0: " << count0 << ", Class 1: " << count1 << "\n\n";
    
    auto start = chrono::high_resolution_clock::now();
    
    RandomForest rf(100, 10, 2, -1, true, 42);
    rf.fit(X, y);
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    
    vector<int> predictions = rf.predict(X);
    
    double endMemory = getMemoryUsageMB();
    
    cout << "\n---------------------------------------------------\n";
    cout << "T_openmp (seconds): " << elapsed.count() << "\n";
    cout << "Memory Usage (MB): " << endMemory << "\n";
    cout << "Threads used: " << num_threads << "\n";
    cout << "---------------------------------------------------\n";
    
    evaluateMetrics(y, predictions);
    cout << "---------------------------------------------------\n";
    
    int correct = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        if (predictions[i] == y[i]) correct++;
    }
    double accuracy = static_cast<double>(correct) / y.size();
    
    ofstream results("results_rf_openmp.csv");
    results << "method,time_seconds,memory_mb,threads,accuracy\n";
    results << "OpenMP RF," << elapsed.count() << "," << endMemory << "," << num_threads << "," << accuracy << "\n";
    results.close();
    
    return 0;
}

