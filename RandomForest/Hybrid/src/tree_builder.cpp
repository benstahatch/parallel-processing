#include <iostream>
#include "tree_builder.h"
#include "bins.h"
#include <algorithm>
#include <omp.h>

// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------
DecisionTreeBuilder::DecisionTreeBuilder(
    const vector<vector<uint8_t>>& X_bins_,
    const vector<int>& labels_,
    const TreeConfig& cfg_
) : X_bins(X_bins_), y(labels_), cfg(cfg_), rng(cfg_.seed) {}

// ------------------------------------------------------------
// Count majority class for leaf node
// ------------------------------------------------------------
int DecisionTreeBuilder::mostCommonClass(const vector<int>& node_indices)
{
    vector<int> counts(cfg.numClasses, 0);
    for (int idx : node_indices) {
        counts[y[idx]]++;
    }
    int bestClass = 0;
    int bestCount = 0;
    for (int c = 0; c < cfg.numClasses; ++c) {
        if (counts[c] > bestCount) {
            bestCount = counts[c];
            bestClass = c;
        }
    }
    return bestClass;
}

// ------------------------------------------------------------
// Create leaf node
// ------------------------------------------------------------
TreeNode* DecisionTreeBuilder::createLeafNode(const vector<int>& node_indices)
{
    TreeNode* node = new TreeNode();
    node->isLeaf = true;
    node->predictedClass = mostCommonClass(node_indices);
    return node;
}

// ------------------------------------------------------------
// Random feature subset sampling
// ------------------------------------------------------------
vector<int> DecisionTreeBuilder::sampleFeatures(int n_features)
{
    vector<int> ft(n_features);
    for (int i = 0; i < n_features; ++i) ft[i] = i;

    shuffle(ft.begin(), ft.end(), rng);

    int k = min(cfg.maxFeatures, n_features);
    return vector<int>(ft.begin(), ft.begin() + k);
}

// ------------------------------------------------------------
// Recursive node building
// ------------------------------------------------------------
TreeNode* DecisionTreeBuilder::buildNode(
    const vector<int>& node_indices,
    int depth
)
{
    int n_samples = node_indices.size();
    int n_features = X_bins.size();

    // Stopping conditions
    if (depth >= cfg.maxDepth ||
        n_samples < cfg.minSamplesSplit)
    {
        return createLeafNode(node_indices);
    }

    // Random feature subset
    vector<int> features = sampleFeatures(n_features);

    // Best split tracking
    int bestFeature = -1;
    int bestBin = -1;
    double bestGain = -1.0;

    // Parallel over features using OpenMP
    #pragma omp parallel
    {
        Histogram hist;
        hist.numBins = NUM_BINS;
        hist.numClasses = cfg.numClasses;
        hist.binCounts.assign(NUM_BINS, vector<int>(cfg.numClasses, 0));

        int localBestFeature = -1;
        int localBestBin = -1;
        double localBestGain = -1.0;

        #pragma omp for schedule(static)
        for (int k = 0; k < features.size(); ++k) {
            int f = features[k];

            // Build histogram for this feature
            buildFeatureHistogram(
                X_bins[f], y, node_indices, hist
            );

            // Find best split
            int binSplit;
            double gain;
            findBestSplitFromHistogram(hist, binSplit, gain);

            if (gain > localBestGain) {
                localBestGain = gain;
                localBestBin = binSplit;
                localBestFeature = f;
            }
        }

        // Reduce to global best
        #pragma omp critical
        {
            if (localBestGain > bestGain) {
                bestGain = localBestGain;
                bestFeature = localBestFeature;
                bestBin = localBestBin;
            }
        }
    }

    // No good split → leaf (allow very small gains for better accuracy)
    if (bestFeature == -1 || bestGain < -0.0001) {
        return createLeafNode(node_indices);
    }

    // Partition indices into left/right
    vector<int> left_idx;
    vector<int> right_idx;
    left_idx.reserve(node_indices.size());
    right_idx.reserve(node_indices.size());

    for (int idx : node_indices) {
        uint8_t bin = X_bins[bestFeature][idx];
        if (bin <= bestBin)
            left_idx.push_back(idx);
        else
            right_idx.push_back(idx);
    }

    // Create node and recurse
    TreeNode* node = new TreeNode();
    node->featureIndex = bestFeature;
    node->binThreshold = bestBin;

    node->left = buildNode(left_idx, depth + 1);
    node->right = buildNode(right_idx, depth + 1);

    return node;
}

// ------------------------------------------------------------
// Public function: build tree from bootstrap sample
// ------------------------------------------------------------
TreeNode* DecisionTreeBuilder::buildTree(const vector<int>& bootstrap_indices)
{
    return buildNode(bootstrap_indices, 0);
}