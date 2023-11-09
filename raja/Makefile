#===============================================================================
# User Options
#===============================================================================

RAJA_PATH ?= 
Umpire_PATH ?=
camp_PATH ?=
TARGET ?= 
SM_VERSION ?=
OFFLOAD_ARCH ?=

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
