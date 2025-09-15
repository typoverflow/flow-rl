
#ifndef FLOWRL_DATA_STRUCTURE_SUMTREE
#define FLOWRL_DATA_STRUCTURE_SUMTREE

#include <cmath>
#include <stdexcept>
#include <vector>
#include <pybind11/pybind11.h>
#include <utility>
#include <algorithm>

class SumTree {
public:
    SumTree(int max_size): max_size(max_size), valid_size(0), curr(0) {
        if (max_size <= 0) throw std::invalid_argument("`max_size` of the sum tree cannot be zero!");
        tree_depth = ceil(log2(this->max_size));
        tree_size = pow(2, tree_depth+1) - 1;
        node_size = pow(2, tree_depth) - 1;
        tree_body.assign(tree_size, 0);
    }
    SumTree& reset() {
        curr = 0;
        valid_size = 0;
        tree_body.assign(tree_size, 0);
        return *this;
    }
    SumTree& update(int idx, double new_value) {
        int tidx = _get_tree_idx(idx);
        double diff = new_value - tree_body[tidx];
        tree_body[tidx] += diff;
        for (; (tidx-1) >= 0; ){
            tidx = _father(tidx);
            tree_body[tidx] += diff;
        }
        return *this;
    }
    SumTree& update(vector<int> idx, vector<double> new_value) {
        for (vector<int>::size_type i=0; i != idx.size(); ++i) {
            update(idx[i], new_value[i]);
        }
        return *this;
    }
    SumTree& add(double new_value) {
        valid_size = std::min(valid_size+1, max_size);
        update(curr, new_value);
        curr = (curr+1) % max_size;
        return *this;
    }
    SumTree& add(std::vector<double> new_values) {
        for (auto value: new_values) {
            add(value);
        }
        return *this;
    }
    std::pair<int, double> find(double target, bool scale=true) {
        if (scale) target *= total();
        int tidx;
        double tvalue;
        tidx = _find_helper(target, 0, tvalue);
        return make_pair(_get_idx(tidx), tvalue);
    }
    std::pair<std::vector<int>, std::vector<double>> find(std::vector<double> target, bool scale=true) {
        std::vector<int> idx;
        std::vector<double> pvalue;
        for (auto t: target) {
            std::pair<int, double> ret = find(t, scale);
            idx.push_back(ret.first);
            pvalue.push_back(ret.second);
        }
        return make_pair(idx, pvalue);
    }
    void show() {
        for (int d=0; d<=tree_depth; ++d) {
            printf("[Depth %d]: ", d);
            for (int i=0; i<std::pow(2, d); ++i)
                printf("%.3f  ", tree_body[i+std::pow(2, d)-1]);
            printf(" \n");
        }
    }
    std::vector<double> values() {
        return values(0, this->valid_size);
    }

    std::vector<double> values(int start, int end) {
        if (end > this->valid_size) end = this->valid_size;
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
    inline double total() {return tree_body[0];}
    inline double min() {return *std::min_element(tree_body.begin()+_get_tree_idx(0), tree_body.begin()+_get_tree_idx(valid_size));}
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

    int _find_helper(double target, int tidx, double& value) {
        if (_left(tidx) >= tree_size) {
            value = tree_body[tidx];
            return tidx;
        }
        double left_value = tree_body[_left(tidx)];
        if (target <= left_value) return _find_helper(target, _left(tidx), value);
        else return _find_helper(target - left_value, _right(tidx), value);
    }
};

#endif
