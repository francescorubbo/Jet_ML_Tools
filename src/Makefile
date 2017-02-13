# Makefile to compile Events.cc

CXX=g++
ifeq ($(shell uname), Darwin)
  CXX = clang++
endif

CXXFLAGS = -O2 -Wall -std=c++0x

INCLUDE   += -DHAVEBOOST
LIBRARIES += -lboost_iostreams

FASTJETINC = $(shell fastjet-config --cxxflags)
PYTHIA8INC = $(shell pythia8-config --cxxflags)

FASTJETLIB = $(shell fastjet-config --libs)
PYTHIA8LIB = $(shell pythia8-config --libs)
LIBRARIES += $(FASTJETLIB) $(PYTHIA8LIB)
INCLUDE   += $(FASTJETINC) $(PYTHIA8INC)


events: Events.o CmdLine.o CleverStream.o
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LIBRARIES)

Events.o: Events.cc
	$(CXX) $(INCLUDE) -o $@ $< -c $(CXXFLAGS)

CmdLine.o: CmdLine.cc CmdLine.hh
	$(CXX) -o $@ $< -c $(CXXFLAGS)

CleverStream.o: CleverStream.cc CleverStream.hh
	$(CXX) -o $@ $< -c $(CXXFLAGS)

clean :
	rm -fv events *.o
