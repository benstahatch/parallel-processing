# ============================
# Makefile for Random Forest
# MPI versions
# ============================

CXX = g++
MPICXX = mpicxx

CXXFLAGS = -O3 -std=c++17
MPIFLAGS = -O3 -std=c++17

MPI_SRC    = random_forest_mpi.cpp

MPI_EXE    = rf_mpi

# -------- Targets ----------
all:  $(MPI_EXE)
$(MPI_EXE): $(MPI_SRC)
	$(MPICXX) $(MPIFLAGS) $< -o $@

clean:
	rm -f $(MPI_EXE)

.PHONY: all clean

