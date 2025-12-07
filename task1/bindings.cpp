#include "gauss_jordan.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(gauss_jordan_cpp, m) {
    m.doc() = "gauss jordan on C++";

    py::class_<GaussJordan>(m, "gauss_jordan")
    .def_static("solve_linear_system", &GaussJordan::solve_linear_system);
}