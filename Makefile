.PHONY: all clean help info

SOURCEDIR = ./src
BINDIR    = ./bin

SOURCES  := $(wildcard $(SOURCEDIR)/*.cpp)

SRC_EXCL  =  $(SOURCEDIR)/BayesRRm_mt.cpp
SRC_EXCL +=  $(SOURCEDIR)/mk_lut.cpp

SOURCES  := $(filter-out $(SRC_EXCL),$(SOURCES))

CXXFLAGS  = -Ofast
#CXXFLAGS  = -O1
CXXFLAGS += -g
CXXFLAGS += -std=c++17
#CXXFLAGS += -D USE_MPI

INCLUDE   = -I$(SOURCEDIR)
INCLUDE += -I/mnt/nfs/clustersw/Debian/stretch/eigen/3.3.7/Eigen
INCLUDE  += -I$(BOOST_ROOT)/include

$(info $$INCLUDE is [${INCLUDE}])

EXEC     ?= hydra_i
CXX       = mpicc
BUILDDIR  = build_intel
CXXFLAGS += -qopenmp
#CXXFLAGS += -xCORE-AVX512 -qopt-zmm-usage=high
CXXFLAGS += -gxx-name=/usr/bin/g++-6    #icc 18 support gcc 4.3 to 6.3.
CXXFLAGS += -U__linux__ -U__USE_MISC -U__NetBSD__ -D__FreeBSD__ -U__PURE_INTEL_C99_HEADERS__ #taking care of math.h problem
CXXFLAGS += -xCORE-AVX2, -axCORE-AVX512 -qopt-zmm-usage=high

OBJ      := $(patsubst $(SOURCEDIR)/%.cpp,$(BUILDDIR)/%.o,$(SOURCES))

LIBS      = -lz


all: dir $(BINDIR)/$(EXEC)

$(BINDIR)/$(EXEC): $(OBJ)
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@

$(OBJ): $(BUILDDIR)/%.o : $(SOURCEDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

dir:
	mkdir -p $(BUILDDIR)
	mkdir -p $(BINDIR)

clean:
	rm -vf $(BUILDDIR)/*.o $(BINDIR)/$(EXEC)

help:
	@echo "Usage: make [ all | clean | help ]"
