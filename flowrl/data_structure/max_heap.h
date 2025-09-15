#ifndef FLOWRL_DATA_STRUCTURE_HEAP
#define FLOWRL_DATA_STRUCTURE_HEAP

#include <cmath>
#include <stdexcept>
#include <vector>
#include <pybind11/pybind11.h>

template<class T>
class Node{
public:
    Node(int value, T data): value(value), data(data) {;}

private:
    double value;
    T data;
}



class MaxHeap {
public:
    MaxHeap(int max_size): max_size(max_size) {
        if (max_size<=0) throw std::invalid_argument("`max_size` of the MaxHeap structure should be greater than zero, dfound %s", max_size);
        heap_depth = ceil(log2(this->max_size));

    }

private:
    int heap_size = 0;
    int heap_depth = 0;
    int curr = 0;
    int valid_size = 0
}

#endif
