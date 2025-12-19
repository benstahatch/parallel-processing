#include "dataset.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>



// loadCSV:
// - reads the CSV file line by line
// - converts strings to doubles
// - separates features from labels
// - populates X and y

void loadCSV(const std::string& filename, FeatureMatrix& X, Labels& y)
{
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file" << filename << std::endl;
        return;
    }

std::string line;
bool skipHeader = true;

    while(std::getline(file, line)) {

        //skip header row
        if (skipHeader) {
        skipHeader = false;
        continue;
        }

        std::stringstream ss(line);
        std::string value;
        FeatureVector row;

        // read comma separated values
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stod(value));
        }

        // last value is the label
        int label = static_cast<int>(row.back());
        row.pop_back();

        X.push_back(row);
        y.push_back(label);
   
    }

    file.close();
}


// normalize each feature column to a range of [0,1]
void minMaxScale(FeatureMatrix& X)
{
    if (X.empty()) return;

    const size_t numFeatures = X[0].size();

    for (size_t j = 0; j < numFeatures; ++j) {

        double minVal = X[0][j];
        double maxVal = X[0][j];

        // find min and max for column j
        for (size_t i = 0; i < X.size(); ++i) {
            minVal = std::min(minVal, X[i][j]);
            maxVal = std::max(maxVal, X[i][j]);
        }

        // scale column j
        for (size_t i = 0; i < X.size(); ++i) {
            if (maxVal != minVal) {
                X[i][j] = (X[i][j] - minVal) / (maxVal - minVal);
            } else {
                X[i][j] = 0.0;
            }
        }

    }


}


