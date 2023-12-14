#===============================================================================
# User Options
#===============================================================================

RAJA_PATH ?= 
Umpire_PATH ?=
camp_PATH ?=
TARGET ?= # Choose Target Backend (CUDA, HIP, OpenMP) Default is Sequential
SM_VERSION ?= # For CUDA Target
OFFLOAD_ARCH ?= # For HIP Target

OPTIMIZE ?= yes
DEBUG ?= no
PROFILE ?= no
MPI ?= no

RAJA_INCLUDE_DIR = $(RAJA_PATH)/include
RAJA_LIB_DIR = $(RAJA_PATH)/lib

Umpire_INCLUDE_DIR = $(Umpire_PATH)/include
Umpire_LIB_DIR = $(Umpire_PATH)/lib

camp_INCLUDE_DIR ?= $(camp_PATH)/include
camp_LIB_DIR = $(camp_PATH)/lib

RAJA_COMPILER_FLAGS ?=
RAJA_LINKER_FLAGS ?=

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

CXXFLAGS := -std=c++14

# Linker Flags
LDFLAGS = -lm

# Debug Flags
ifeq ($(DEBUG),yes)
  CXXFLAGS += -g -G
  LDFLAGS += -g -G
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

ifeq ($(TARGET),CUDA)
  CXX = nvcc
  RAJA_COMPILER_FLAGS = -x cu --expt-relaxed-constexpr -forward-unknown-to-host-compiler -extended-lambda -restrict -keep -arch=sm_$(SM_VERSION) 
  CXXFLAGS += $(CUDAFLAGS)
  RAJA_LINKER_FLAGS += -Xlinker -rpath=$(RAJA_LIB_DIR) -Xlinker -rpath=$(Umpire_LIB_DIR)
endif  

ifeq ($(TARGET),HIP)
  CXX = hipcc
  RAJA_COMPILER_FLAGS = --offload-arch=$(OFFLOAD_ARCH)
  CXXFLAGS += $(HIPFLAGS)
  RAJA_LINKER_FLAGS += -Xlinker -rpath=$(RAJA_LIB_DIR) -Xlinker -rpath=$(Umpire_LIB_DIR)
endif

ifeq ($(TARGET), OpenMP)
  CXXFLAGS += -fopenmp
  LDFLAGS += -fopenmp
  LDFLAGS += -Xlinker -rpath=$(RAJA_LIB_DIR) -Xlinker -rpath=$(Umpire_LIB_DIR)
endif

default: $(program)

RAJA_COMPILER_FLAGS += -I./ -I$(RAJA_INCLUDE_DIR) -I$(Umpire_INCLUDE_DIR) -I$(camp_INCLUDE_DIR)
RAJA_LINKER_FLAGS += -lRAJA -lumpire -lcamp
RAJA_LINKER_FLAGS += -L$(RAJA_LIB_DIR) -L$(Umpire_LIB_DIR) -L$(camp_LIB_DIR)

CXXFLAGS += -w

$(program): $(obj) XSbench_header.h Makefile
	$(CXX) $(obj) -o $@ $(LDFLAGS) $(RAJA_LINKER_FLAGS)

%.o: %.cpp XSbench_header.h Makefile
	$(CXX) $(CXXFLAGS) $(RAJA_COMPILER_FLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) *.cpp1* *.cpp4* *.ptx *.cubin *.cudafe1* *.fatbin* *.module_id

edit:
	vim -p $(source) XSbench_header.h

run:
	./$(program)


