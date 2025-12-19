#include "mpi_forest.h"
#include <iostream>
#include <algorithm>
using namespace std;

// ------------------------------------------------------------
// Constructor: determine tree split across ranks
// ------------------------------------------------------------
MPIRandomForest::MPIRandomForest(
    const vector<vector<uint8_t>>& X_bins_,
    const vector<int>& labels_,
    const ForestConfig& cfg_
)
: X_bins(X_bins_), y(labels_), cfg(cfg_)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Evenly split trees across MPI ranks
    treesPerRank = cfg.nTrees / size;
    startTree    = rank * treesPerRank;
    endTree      = (rank == size - 1)
                     ? cfg.nTrees
                     : (rank + 1) * treesPerRank;

    // Local forest config (each rank trains its subset)
    ForestConfig localCfg = cfg;
    localCfg.nTrees = endTree - startTree;
    localCfg.seed   = cfg.seed + rank * 1000;  // shift seeds per rank

    localForest = new RandomForest(X_bins, y, localCfg);
}

// ------------------------------------------------------------
// Train the local forest with OpenMP
// ------------------------------------------------------------
void MPIRandomForest::fit()
{
    if (rank == 0) {
        cout << "MPI Random Forest - Total Trees: " << cfg.nTrees << "\n";
        cout << "MPI Ranks: " << size << "\n\n";
    }

    // Train only local subset of trees
    localForest->fit();

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        cout << "All MPI ranks completed local training.\n";
}

// ------------------------------------------------------------
// MPI prediction: each rank predicts using local trees,
// then votes are combined using MPI_Allreduce.
// ------------------------------------------------------------
vector<int> MPIRandomForest::predict()
{
    int n_samples = y.size();
    int numClasses = cfg.numClasses;

    // Local predictions
    vector<int> localPreds = localForest->predictBatch();

    // Convert local predictions to one-hot votes per class
    vector<int> localVotes(n_samples * numClasses, 0);

    for (int i = 0; i < n_samples; ++i) {
    int cls = localPreds[i];
    if (cls < 0 || cls >= numClasses) cls = 0;
    localVotes[i * numClasses + cls]++;
}


    // Global votes after MPI reduction
    vector<int> globalVotes(n_samples * numClasses, 0);

    MPI_Allreduce(
        localVotes.data(),
        globalVotes.data(),
        n_samples * numClasses,
        MPI_INT,
        MPI_SUM,
        MPI_COMM_WORLD
    );

    // Final predictions by majority vote
    vector<int> finalPred(n_samples);

    for (int i = 0; i < n_samples; ++i) {
        int bestClass = 0;
        int bestCount = globalVotes[i * numClasses];

        for (int c = 1; c < numClasses; ++c) {
            int count = globalVotes[i * numClasses + c];
            if (count > bestCount) {
                bestCount = count;
                bestClass = c;
            }
        }
        finalPred[i] = bestClass;
    }

    return finalPred;
}
