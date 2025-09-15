#ifndef FLOWRL_DATA_STRUCTURE_MINTREE
#define FLOWRL_DATA_STRUCTURE_MINTREE

#include <cmath>
#include <stdexcept>
#include <vector>
#include <pybind11/pybind11.h>
#include <utility>
#include <climits>

class MinTree {
public:
    MinTree(int max_size): max_size(max_size), valid_size(0), curr(0) {
        if (max_size <= 0) throw std::invalid_argument("`max_size` of the sum tree cannot be zero!");
        tree_depth = ceil(log2(this->max_size));
        tree_size = pow(2, tree_depth+1) - 1;
        node_size = pow(2, tree_depth) - 1;
        tree_body.assign(tree_size, std::numeric_limits<double>::max());
    }
    MinTree& reset() {
        curr = 0;
        valid_size = 0;
        tree_body.assign(tree_size, std::numeric_limits<double>::max());
        return *this;
    }
    MinTree& update(int idx, double new_value) {
        int tidx = _get_tree_idx(idx);
        tree_body[tidx] = new_value;
        for (; (tidx-1) >= 0; ){
            tidx = _father(tidx);
            double old_value = tree_body[tidx];
            double left_value = tree_body[_left(tidx)];
            double right_value = tree_body[_right(tidx)];
            tree_body[tidx] = left_value < right_value? left_value : right_value;
            if (tree_body[tidx] == old_value)
                break;
        }
        return *this;
    }
    MinTree& update(vector<int> idx, vector<double> new_value) {
        for (vector<int>::size_type i=0; i != idx.size(); ++i) {
            update(idx[i], new_value[i]);
        }
        return *this;
    }
    MinTree& add(double new_value) {
        valid_size = std::min(valid_size+1, max_size);
        update(curr, new_value);
        curr = (curr+1) % max_size;
        return *this;
    }
    MinTree& add(std::vector<double> new_values) {
        for (auto value: new_values) {
            add(value);
        }
        return *this;
    }
    void show() {
        for (int d=0; d<=tree_depth; ++d) {
            printf("[Depth %d]: ", d);
            for (int i=0; i<std::pow(2, d); ++i)
                if (tree_body[i+std::pow(2, d)-1] == std::numeric_limits<double>::max())
                    printf("NaN  ");
                else
                    printf("%.3f  ", tree_body[i+std::pow(2, d)-1]);
            printf(" \n");
        }
    }
    std::vector<double> values() {
        return values(0, this->valid_size);
    }

    std::vector<double> values(int start, int end) {
        if (end > this->valid_size) end = this->valid_size;
        printf("%d", this->curr);
        auto vstart = this->tree_body.end() - (tree_size - node_size) + start;
        auto vend = this->tree_body.end() - (tree_size - node_size) + end;
        return std::vector<double>(vstart, vend);
    }
    std::vector<double> values(vector<int> indices) {
        std::vector<double> v;
        auto vstart = this->tree_body.end() - (tree_size - node_size);
        for (auto i: indices) {
            v.push_back(*(vstart+i));
        }
        return v;
    }
    inline double min() {return tree_body[0];}
private:
    int max_size;
    int tree_depth;
    int tree_size;
    int node_size;
    int valid_size = 0;
    int curr = 0;
    std::vector<double> tree_body;

    inline int _left(const int tree_idx) {return 2*tree_idx+1;}
    inline int _right(const int tree_idx) {return 2*tree_idx+2;}
    inline int _father(const int tree_idx) {return (tree_idx-1)/2;}
    inline int _get_tree_idx(const int idx) {return node_size + idx;}
    inline int _get_idx(const int tidx) {return tidx - node_size;}
};

#endif
