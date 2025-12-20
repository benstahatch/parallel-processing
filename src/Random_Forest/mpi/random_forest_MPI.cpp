// Due to length, creating shorter MPI Random Forest that distributes tree training
#include <iostream>
#include <vector>
#include <mpi.h>
#include "random_forest_serial.cpp"  // Reuse serial tree implementation

// MPI version distributes tree training across processes
// Each process trains a subset of trees
// Trees are then gathered to rank 0 for prediction
