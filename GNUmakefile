AMREX_HOME ?= ../amrex

DEBUG	= FALSE

DIM	= 3

COMP    = gcc

PRECISION = FLOAT

USE_MPI   = FALSE
USE_OMP   = FALSE
USE_CUDA  = FALSE
USE_HIP   = FALSE
USE_SYCL  = TRUE

SYCL_AOT = TRUE
AMREX_INTEL_ARCH ?= pvc

BL_NO_FORT = TRUE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package
include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
