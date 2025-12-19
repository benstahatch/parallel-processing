#ifndef BINS_H
#define BINS_H

#include <vector>
#include <algorithm>
#include <cmath>
using namespace std;

// ------------------------------------------------------------
// Binning configuration
const int NUM_BINS = 256;    // 256-bin quantization (0–255)

// ------------------------------------------------------------
// BinMapping: stores min, max, and thresholds for each feature.
// This allows fast float -> bin conversion.
// ------------------------------------------------------------
struct BinMapping {
    double minVal;
    double maxVal;
    double step;
};

// ------------------------------------------------------------
// Compute bin mapping for each feature.
// This is called one time at dataset load.
// ------------------------------------------------------------
vector<BinMapping> computeBinMappings(const vector<vector<double>>& X);

// ------------------------------------------------------------
// Convert the full dataset into uint8 bins.
// Output layout: column-major (faster for histogram building).
// ------------------------------------------------------------
void binarizeDataset(
    const vector<vector<double>>& X,
    const vector<BinMapping>& mappings,
    vector<vector<uint8_t>>& X_bins
);

// ------------------------------------------------------------
// Convert a single floating-point value x into bin 0–255.
// ------------------------------------------------------------
inline uint8_t binValue(double x, const BinMapping& map) {
    if (x <= map.minVal) return 0;
    if (x >= map.maxVal) return NUM_BINS - 1;
    return static_cast<uint8_t>((x - map.minVal) / map.step);
}

#endif