# ------------------------------------------------------------
# Hybrid MPI + OpenMP Random Forest — Makefile
# Intel (x86_64) & Apple Silicon (arm64)
# LLVM / OpenMP / MPI
# ------------------------------------------------------------

# Detect Homebrew prefix (Intel vs Apple Silicon)
HOMEBREW_PREFIX := $(shell brew --prefix 2>/dev/null)

LLVM_CLANG := $(HOMEBREW_PREFIX)/opt/llvm/bin/clang++
LLVM_OMP   := $(HOMEBREW_PREFIX)/opt/llvm/lib

# Force mpicxx to use LLVM clang++
MPI_CXX = OMPI_CXX=$(LLVM_CLANG) mpicxx

# Compiler / linker flags
CXXFLAGS = -O3 -std=c++17 -fopenmp -Iinclude
LDFLAGS  = -L$(LLVM_OMP) -lomp

# OpenMP runtime configuration
export OMP_NUM_THREADS ?= 4

# Target
TARGET = hybrid_rf

# Sources
SRC = src/main.cpp \
      src/bins.cpp \
      src/histogram.cpp \
      src/tree_builder.cpp \
      src/forest.cpp \
      src/mpi_forest.cpp

# ------------------------------------------------------------
# Build
# ------------------------------------------------------------
.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(MPI_CXX) $(CXXFLAGS) $(SRC) $(LDFLAGS) -o $(TARGET)
	@echo "------------------------------------------------------------"
	@echo "Build complete: ./$(TARGET)"
	@echo "OMP_NUM_THREADS=$(OMP_NUM_THREADS)"
	@echo "Use 'make run' to execute with MPI"
	@echo "------------------------------------------------------------"

# ------------------------------------------------------------
# Clean
# ------------------------------------------------------------
clean:
	rm -f $(TARGET)
	@echo "Clean complete."

# ------------------------------------------------------------
# Run (default 4 MPI ranks)
# ------------------------------------------------------------
NP ?= 4

run: $(TARGET)
	mpirun -np $(NP) ./$(TARGET)
