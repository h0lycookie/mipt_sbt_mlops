#pragma once
#include <vector>

class GaussJordan {
public:
    static std::vector<double> solve_linear_system(const std::vector<std::vector<double>> &A, const std::vector<double> &b);

private:
    static std::vector<std::vector<double>> gauss_jordan(const std::vector<std::vector<double>> &matrix);
};
