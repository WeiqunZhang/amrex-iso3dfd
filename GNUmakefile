AMREX_HOME ?= ../amrex

DEBUG	= FALSE

DIM	= 3

COMP    = gcc

PRECISION = FLOAT

USE_MPI   = FALSE
USE_OMP   = FALSE
USE_CUDA  = FALSE
USE_HIP   = FALSE
USE_SYCL  = FALSE

#SYCL_AOT = TRUE
AMREX_INTEL_ARCH ?= pvc
AMREX_CUDA_ARCH ?= 80
AMREX_AMD_ARCH ?= gfx90a

BL_NO_FORT = TRUE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package
include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
