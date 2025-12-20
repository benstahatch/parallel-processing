#include "bins.h"

// ------------------------------------------------------------
// Compute min, max, and bin step for each feature
// ------------------------------------------------------------
vector<BinMapping> computeBinMappings(const vector<vector<double>>& X)
{
    int n_samples = X.size();
    int n_features = X[0].size();

    vector<BinMapping> mappings(n_features);

    for (int f = 0; f < n_features; f++) {
        double mn = X[0][f];
        double mx = X[0][f];

        for (int i = 1; i < n_samples; i++) {
            double v = X[i][f];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }

        if (mx == mn) {
            // Constant feature: avoid division by zero
            mappings[f].minVal = mn;
            mappings[f].maxVal = mx;
            mappings[f].step = 1.0;
        } else {
            mappings[f].minVal = mn;
            mappings[f].maxVal = mx;
            mappings[f].step = (mx - mn) / double(NUM_BINS);
        }
    }

    return mappings;
}

// ------------------------------------------------------------
// Convert full dataset to bins (column-major layout for speed)
// ------------------------------------------------------------
void binarizeDataset(
    const vector<vector<double>>& X,
    const vector<BinMapping>& mappings,
    vector<vector<uint8_t>>& X_bins
)
{
    int n_samples = X.size();
    int n_features = X[0].size();

    X_bins.assign(n_features, vector<uint8_t>(n_samples));

    for (int f = 0; f < n_features; f++) {
        const BinMapping& map = mappings[f];

        for (int i = 0; i < n_samples; i++) {
            X_bins[f][i] = binValue(X[i][f], map);
        }
    }
}
