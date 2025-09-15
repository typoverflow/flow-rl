#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <stdexcept>
#include <vector>
#include <cmath>

#include <iostream>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace std;

#include "sumtree.h"
#include "mintree.h"

PYBIND11_MODULE(data_structure, m) {
    py::class_<SumTree>(m, "SumTree")
        .def(py::init<int>())
        .def("reset", &SumTree::reset)
        .def("update", static_cast<SumTree& (SumTree::*)(int, double)>(&SumTree::update))
        .def("update", static_cast<SumTree& (SumTree::*)(vector<int>, vector<double>)>(&SumTree::update))
        .def("add", static_cast<SumTree& (SumTree::*)(vector<double>)>(&SumTree::add))
        .def("add", static_cast<SumTree& (SumTree::*)(double)>(&SumTree::add))
        .def("find", static_cast<pair<int, double> (SumTree::*)(double, bool)>(&SumTree::find), "Search the tree and return the index with given target value. ",
            "target"_a, "scale"_a=true)
        .def("find", static_cast<pair<vector<int>, vector<double>> (SumTree::*)(vector<double>, bool)>(&SumTree::find), "Search the tree and return the index with given target value in batch. ",
            "target"_a, "scale"_a=true)
        .def("show", &SumTree::show)
        .def("values", static_cast<vector<double> (SumTree::*)()>(&SumTree::values))
        .def("values", static_cast<vector<double> (SumTree::*)(int, int)>(&SumTree::values))
        .def("values", static_cast<vector<double> (SumTree::*)(vector<int>)>(&SumTree::values))
        .def("total", &SumTree::total)
        .def("min", &SumTree::min)
        ;

    py::class_<MinTree>(m, "MinTree")
        .def(py::init<int>())
        .def("reset", &MinTree::reset)
        .def("update", static_cast<MinTree& (MinTree::*)(int, double)>(&MinTree::update))
        .def("update", static_cast<MinTree& (MinTree::*)(vector<int>, vector<double>)>(&MinTree::update))
        .def("add", static_cast<MinTree& (MinTree::*)(vector<double>)>(&MinTree::add))
        .def("add", static_cast<MinTree& (MinTree::*)(double)>(&MinTree::add))
        .def("show", &MinTree::show)
        .def("min", &MinTree::min)
        ;
}
