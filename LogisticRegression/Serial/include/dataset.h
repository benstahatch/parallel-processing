#ifndef DATASET_H
#define DATASET_H

#include "types.h"
#include <string>


void loadCSV(const std::string& filename, FeatureMatrix& X, Labels& y);


void minMaxScale(FeatureMatrix& X);



#endif