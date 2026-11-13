# Build for EPM. Grading uses these flags; do not change them.
# sm_89 targets L4 GPU; requires CUDA 11.8 or newer.
NVCC ?= nvcc
ARCH ?= sm_89
OMP  ?= 1

SRC_CU  = $(wildcard src/*.cu)
SRC_CPP = src/gen.cpp src/ref_cpu.cpp src/verify.cpp
HEADERS = $(wildcard include/*.h) $(wildcard src/*.cuh)

NVFLAGS = -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Iinclude
ifeq ($(OMP),1)
NVFLAGS += -Xcompiler -fopenmp -lgomp
endif

epm: $(SRC_CU) $(SRC_CPP) $(HEADERS)
	$(NVCC) $(NVFLAGS) -o $@ $(SRC_CU) $(SRC_CPP)

clean:
	rm -f epm

.PHONY: clean
