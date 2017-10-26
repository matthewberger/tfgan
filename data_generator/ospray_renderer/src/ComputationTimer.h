#ifndef COMPUTATIONTIMER_H
#define COMPUTATIONTIMER_H

#include <time.h>
#include <iostream>

using namespace std;

class ComputationTimer {
public:
    ComputationTimer(string _computation);

    ~ComputationTimer();

    void start() { computation_start = clock(); }

    void end() {
        computation_end = clock();
        total_time = ((double) (computation_end - computation_start)) / CLOCKS_PER_SEC;
    }

    void setName(string name){computation = name;}

    string getComputation() { return computation; }

    double getElapsedTime() { return total_time; }

    void dump_time() { cout << computation << " : " << total_time << " s " << endl; }

private:
    clock_t computation_start, computation_end;
    double total_time;
    string computation;
};

#endif
