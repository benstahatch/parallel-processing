#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <vector>
#include <cstdint>
using namespace std;

// ------------------------------------------------------------
// Histogram structure for one feature at a tree node.
// For each bin (0-255), store class counts.
// ------------------------------------------------------------
struct Histogram {
    vector<vector<int>> binCounts; // [num_bins][num_classes]
    int numBins;
    int numClasses;
};

// ------------------------------------------------------------
// Build histogram for a single feature at a node.
// X_bins[j] is the column of binned feature values.
// node_indices: rows belonging to this node
// ------------------------------------------------------------
void buildFeatureHistogram(
    const vector<uint8_t>& featureCol,
    const vector<int>& labels,
    const vector<int>& node_indices,
    Histogram& hist
);

// ------------------------------------------------------------
// Compute best split from histogram using Gini impurity.
// Returns best bin index and best Gini gain.
// ------------------------------------------------------------
void findBestSplitFromHistogram(
    const Histogram& hist,
    int& bestBin,
    double& bestGain
);

// ------------------------------------------------------------
// Compute Gini impurity given class counts.
// ------------------------------------------------------------
double giniFromCounts(const vector<int>& counts);

#endif