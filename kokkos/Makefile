#===============================================================================
# User Options
#===============================================================================

KOKKOS_PATH ?= /global/homes/j/jhdavis/XSBench/kokkos/kokkos
CUDA_PATH   ?= ${CUDA_HOME}
COMPILER    ?= gnu
OPTIMIZE    ?= yes
DEBUG       ?= no
PROFILE     ?= no
MPI         ?= no
KOKKOS_DEVICES ?= OpenMP,Cuda
KOKKOS_ARCH    ?= Zen3,Ampere80
SM_VERSION  ?= 80
NVTX        ?= no
LINEINFO    ?= no

#===============================================================================
# Program name & source code list
#===============================================================================

program = XSBench

source = \
Main.cpp \
io.cpp \
Simulation.cpp \
GridInit.cpp \
XSutils.cpp \
Materials.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CXXFLAGS := -Wall

# Linker Flags
LDFLAGS = -lm

# NVIDIA Compiler
ifeq ($(COMPILER),nvidia)
  CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
  CXXFLAGS += -ccbin nvc++ -Xcompiler -Wall -Xcompiler -O3 -arch=sm_$(SM_VERSION)
endif

# Clang Compiler
ifeq ($(COMPILER),llvm)
  CXX = clang++
  CXXFLAGS += -flto -fopenmp -DOPENMP
endif

# GCC Compiler
ifeq ($(COMPILER),gnu)
  ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
    CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
    CXXFLAGS += -ccbin g++ -Xcompiler -Wall -Xcompiler -O3 -Xcompiler -fopenmp -DOPENMP
  else
    CXX = g++
    CXXFLAGS += -flto -fopenmp -DOPENMP
  endif
endif


# Debug Flags
ifeq ($(DEBUG),yes)
  CXXFLAGS += -g
  LDFLAGS  += -g
ifeq ($(COMPILER),nvidia)
  CXXFLAGS += -G
  LDFLAGS  += -G
endif
endif

# Profiling Flags
ifeq ($(PROFILE),yes)
  CXXFLAGS += -pg
  LDFLAGS  += -pg
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CXXFLAGS += -O3
endif

# MPI
ifeq ($(MPI),yes)
  CXX = mpicxx
  CXXFLAGS += -DMPI
endif

# Enable lineinfo without disabling GPU optimizations
ifeq ($(LINEINFO),yes)
  CXXFLAGS += -g
  LDFLAGS  += -g
ifeq ($(COMPILER),nvidia)
  CXXFLAGS += -lineinfo
  LDFLAGS  += -lineinfo
endif
endif

# Enable NVTX annotations
ifeq ($(NVTX),yes)
	CXXFLAGS += -DUSE_NVTX
endif

#===============================================================================
# Targets to Build
#===============================================================================

default: $(program)

include $(KOKKOS_PATH)/Makefile.kokkos

$(program): $(obj) XSbench_header.hpp Makefile $(KOKKOS_LINK_DEPENDS)
	$(CXX) $(CXXFLAGS) $(KOKKOS_LDFLAGS) $(obj) $(KOKKOS_LIBS) -o $@ $(LDFLAGS)

%.o: %.cpp XSbench_header.hpp Makefile $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) *.tmp Kokkos* desul/ libkokkos.a Lock_Array_CUDA.o

edit:
	vim -p $(source) XSbench_header.hpp

run:
	./$(program)
