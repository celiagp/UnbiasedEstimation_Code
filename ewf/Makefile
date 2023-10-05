CC := g++
CFLAGS := -Wall -Wextra -std=c++11 -O3 # -g
LDFLAGS := -I/usr/local/lib/boost_1_78_0 -I/home/celitona/Documents/git/EWF /usr/local/lib/libconfig++.a

#Make sure LDFLAGS points towards the locations where:

# 1. your boost library resides(e.g.'/home/ubuntu/Desktop/boost_1_78_0')
# 2. the source files for the program reside(e.g.'/home/ubuntu/Desktop/EWF')
# 3. the library libconfig ++.a resides(e.g.'/usr/local/lib/libconfig++.a')

PROGRAMS := main

all: $(PROGRAMS)

main: main.cpp WrightFisher.cpp myHelpers.cpp Polynomial.cpp PolynomialRootFinder.cpp
	$(CC)   $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean: rm -f *.o $(PROGRAMS)
