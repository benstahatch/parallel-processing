#include <iostream>
#include "histogram.h"
#include <algorithm>
#include <numeric>

// ------------------------------------------------------------
// Compute Gini impurity for class counts.
// Same concept as your previous giniImpurity(), but now uses counts.
// ------------------------------------------------------------
double giniFromCounts(const vector<int>& counts)
{
    int total = 0;
    for (int c : counts) total += c;
    if (total == 0) return 0.0;

    double gini = 1.0;
    for (int c : counts) {
        double p = (double)c / total;
        gini -= p * p;
    }
    return gini;
}

// ------------------------------------------------------------
// Build histogram for one feature at a node.
// binCounts[b][class] = count of samples in bin b of this class
// ------------------------------------------------------------
void buildFeatureHistogram(
    const vector<uint8_t>& featureCol,
    const vector<int>& labels,
    const vector<int>& node_indices,
    Histogram& hist
)
{
    // Reset histogram counts
    for (int b = 0; b < hist.numBins; ++b) {
        std::fill(hist.binCounts[b].begin(), hist.binCounts[b].end(), 0);
    }

    // Accumulate counts
    for (int row : node_indices) {
        uint8_t bin = featureCol[row];
        int y = labels[row];
        hist.binCounts[bin][y]++;
    }
}

// ------------------------------------------------------------
// Fast split search using histogram bins.
// Left = bins <= b
// Right = bins > b
// ------------------------------------------------------------
void findBestSplitFromHistogram(
    const Histogram& hist,
    int& bestBin,
    double& bestGain
)
{
    int B = hist.numBins;
    int C = hist.numClasses;

    vector<int> leftCounts(C, 0);
    vector<int> rightCounts(C, 0);

    // Initialize rightCounts with total sample counts
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            rightCounts[c] += hist.binCounts[b][c];
        }
    }

    // Compute parent impurity ONCE
    vector<int> parentCounts(C, 0);
    for (int c = 0; c < C; ++c) {
        parentCounts[c] = rightCounts[c];
    }
    double parentImp = giniFromCounts(parentCounts);

    bestGain = -1.0;
    bestBin  = -1;

    // Scan bins as split points
    for (int b = 0; b < B - 1; ++b) {

        // Move bin b from right to left
        for (int c = 0; c < C; ++c) {
            int val = hist.binCounts[b][c];
            leftCounts[c]  += val;
            rightCounts[c] -= val;
        }

        // Compute sizes
        int totalLeft  = 0;
        int totalRight = 0;
        for (int c = 0; c < C; ++c) {
            totalLeft  += leftCounts[c];
            totalRight += rightCounts[c];

            
        }
        int total = totalLeft + totalRight;
        if (totalLeft == 0 || totalRight == 0) continue;

        // Child impurities
        double g_left  = giniFromCounts(leftCounts);
        double g_right = giniFromCounts(rightCounts);

        // Weighted impurity
        double weighted =
            (totalLeft  / (double)total) * g_left +
            (totalRight / (double)total) * g_right;

        // Gini gain
        double gain = parentImp - weighted;

        if (gain > bestGain) {
            bestGain = gain;
            bestBin  = b;
        }
    }
}
