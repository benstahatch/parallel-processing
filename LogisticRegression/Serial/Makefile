# ------------------------------------------------------------
# Serial Logistic Regression — Makefile
# macOS (Intel & Apple Silicon), Linux, Windows (MinGW / WSL)
# C++17, Portable Serial Build
# ------------------------------------------------------------

# Compiler (can be overridden: make CXX=clang++)
CXX ?= g++

# Compiler flags
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra -Iinclude

# Target executable
TARGET = serial_lr

# Source files
SRC = src/main.cpp \
      src/math_utils.cpp \
      src/logistic_regression.cpp \
      src/metrics.cpp \
      src/dataset.cpp \
      src/performance.cpp \
      src/output.cpp

# Object files
OBJ = $(SRC:.cpp=.o)

# ------------------------------------------------------------
# Platform-specific system libraries
# ------------------------------------------------------------
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Linux)
    SYS_LIBS =
endif

ifeq ($(UNAME_S),Darwin)
    SYS_LIBS =
endif

ifeq ($(OS),Windows_NT)
    SYS_LIBS = -lpsapi
endif

# ------------------------------------------------------------
# Build targets
# ------------------------------------------------------------
.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) $(SYS_LIBS) -o $(TARGET)
	@echo "------------------------------------------------------------"
	@echo "Build complete: ./$(TARGET)"
	@echo "------------------------------------------------------------"

# ------------------------------------------------------------
# Compile objects
# ------------------------------------------------------------
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
run: $(TARGET)
	./$(TARGET)

# ------------------------------------------------------------
# Clean
# ------------------------------------------------------------
clean:
	rm -f $(OBJ) $(TARGET)
	@echo "Clean complete."
