# ------------------------------------------------------------
# Hybrid MPI + OpenMP Random Forest — Makefile (macOS)
# ------------------------------------------------------------

# Use system clang++ with Homebrew libomp
CXX = clang++
OMP_LIB = /opt/homebrew/opt/libomp/lib

# Use mpicxx directly
MPI_CXX = mpicxx

CXXFLAGS = -O3 -std=c++17 -march=native -Xpreprocessor -fopenmp -Iinclude -I/opt/homebrew/opt/libomp/include
LDFLAGS  = -L$(OMP_LIB) -lomp

# Targets
SERIAL_FOREST_TARGET = serial_forest
FOREST_TARGET = forest
MPI_FOREST_TARGET = mpi_forest
HISTOGRAM_TEST_TARGET = test_histogram
TREE_BUILDER_TEST_TARGET = test_tree_builder

# Common source files
COMMON_SRC = src/bins.cpp \
             src/histogram.cpp \
             src/tree_builder.cpp \
             src/forest.cpp

# Serial Forest source (no parallelization)
SERIAL_FOREST_SRC = src/main_serial.cpp \
                    src/bins.cpp \
                    src/histogram.cpp \
                    src/tree_builder.cpp

# Forest-only source (OpenMP, no MPI)
FOREST_SRC = src/main_forest.cpp \
             $(COMMON_SRC)

# MPI Forest source
MPI_FOREST_SRC = src/main_mpi_forest.cpp \
                 $(COMMON_SRC) \
                 src/mpi_forest.cpp

# Histogram test source
HISTOGRAM_TEST_SRC = src/test_histogram.cpp \
                     src/bins.cpp \
                     src/histogram.cpp

# Tree builder test source
TREE_BUILDER_TEST_SRC = src/test_tree_builder.cpp \
                        src/bins.cpp \
                        src/histogram.cpp \
                        src/tree_builder.cpp

# ------------------------------------------------------------
# Build all executables
# ------------------------------------------------------------
all: $(SERIAL_FOREST_TARGET) $(FOREST_TARGET) $(MPI_FOREST_TARGET) $(HISTOGRAM_TEST_TARGET) $(TREE_BUILDER_TEST_TARGET)

# Serial Forest (no parallelization - compiled with OpenMP but runs with 1 thread)
$(SERIAL_FOREST_TARGET): $(SERIAL_FOREST_SRC)
	$(CXX) $(CXXFLAGS) $(SERIAL_FOREST_SRC) $(LDFLAGS) -o $(SERIAL_FOREST_TARGET)
	@echo "Built: ./$(SERIAL_FOREST_TARGET) (Serial - no parallelization)"

# Regular Forest (OpenMP only)
$(FOREST_TARGET): $(FOREST_SRC)
	$(CXX) $(CXXFLAGS) $(FOREST_SRC) $(LDFLAGS) -o $(FOREST_TARGET)
	@echo "Built: ./$(FOREST_TARGET) (OpenMP only)"

# MPI Forest (MPI + OpenMP)
$(MPI_FOREST_TARGET): $(MPI_FOREST_SRC)
	$(MPI_CXX) $(CXXFLAGS) $(MPI_FOREST_SRC) $(LDFLAGS) -o $(MPI_FOREST_TARGET)
	@echo "Built: ./$(MPI_FOREST_TARGET) (MPI + OpenMP)"

# Histogram Test
$(HISTOGRAM_TEST_TARGET): $(HISTOGRAM_TEST_SRC)
	$(CXX) $(CXXFLAGS) $(HISTOGRAM_TEST_SRC) $(LDFLAGS) -o $(HISTOGRAM_TEST_TARGET)
	@echo "Built: ./$(HISTOGRAM_TEST_TARGET)"

# Tree Builder Test
$(TREE_BUILDER_TEST_TARGET): $(TREE_BUILDER_TEST_SRC)
	$(CXX) $(CXXFLAGS) $(TREE_BUILDER_TEST_SRC) $(LDFLAGS) -o $(TREE_BUILDER_TEST_TARGET)
	@echo "Built: ./$(TREE_BUILDER_TEST_TARGET)"

# ------------------------------------------------------------
# Clean
# ------------------------------------------------------------
clean:
	rm -f $(FOREST_TARGET) $(MPI_FOREST_TARGET) $(HISTOGRAM_TEST_TARGET) $(TREE_BUILDER_TEST_TARGET)
	@echo "Clean complete."

# ------------------------------------------------------------
# Run targets
# ------------------------------------------------------------
NP ?= 2

run-forest: $(FOREST_TARGET)
	./$(FOREST_TARGET)

run-mpi: $(MPI_FOREST_TARGET)
	mpirun -np $(NP) ./$(MPI_FOREST_TARGET)

run-histogram: $(HISTOGRAM_TEST_TARGET)
	./$(HISTOGRAM_TEST_TARGET)

run-tree-builder: $(TREE_BUILDER_TEST_TARGET)
	./$(TREE_BUILDER_TEST_TARGET)

# Visualization - Compare Models
visualize: 
	@echo "Running models and generating comparison visualizations..."
	@python3 -c "import matplotlib, numpy" 2>/dev/null && \
		python3 compare_models.py || \
		echo "Error: Please install matplotlib and numpy: pip3 install --break-system-packages matplotlib numpy"

# Default run (MPI forest)
run: run-mpi

.PHONY: all clean run run-forest run-mpi run-histogram run-tree-builder visualize
