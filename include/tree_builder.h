#ifndef TREE_BUILDER_H
#define TREE_BUILDER_H

#include <vector>
#include <cstdint>
#include <random>
#include "histogram.h"
using namespace std;

// ------------------------------------------------------------
// Tree node structure (similar to your original code).
// Uses bin index (0–255) instead of float threshold.
// ------------------------------------------------------------
struct TreeNode {
    bool isLeaf;
    int predictedClass;
    int featureIndex;
    uint8_t binThreshold;
    TreeNode* left;
    TreeNode* right;

    TreeNode() :
        isLeaf(false), predictedClass(0),
        featureIndex(-1), binThreshold(0),
        left(nullptr), right(nullptr) {}
};

// ------------------------------------------------------------
// Tree training configuration
// ------------------------------------------------------------
struct TreeConfig {
    int maxDepth;
    int minSamplesSplit;
    int maxFeatures;
    int numClasses;
    int seed;
};

// ------------------------------------------------------------
// Decision tree builder using:
// - binned dataset (column-major)
// - index-based node construction
// - OpenMP split parallelism
// ------------------------------------------------------------
class DecisionTreeBuilder {
private:
    const vector<vector<uint8_t>>& X_bins;
    const vector<int>& y;
    TreeConfig cfg;
    mt19937 rng;

    // Create leaf node
    TreeNode* createLeafNode(const vector<int>& node_indices);

    // Count majority class
    int mostCommonClass(const vector<int>& node_indices);

    // Build tree recursively
    TreeNode* buildNode(
        const vector<int>& node_indices,
        int depth
    );

    // Random subset of feature indices
    vector<int> sampleFeatures(int n_features);

public:
    DecisionTreeBuilder(
        const vector<vector<uint8_t>>& X_bins,
        const vector<int>& labels,
        const TreeConfig& cfg
    );

    TreeNode* buildTree(const vector<int>& bootstrap_indices);
};

#endif