#include "gauss_jordan.h"
#include <stdexcept>
#include <vector>

std::vector<double> GaussJordan::solve_linear_system(
    const std::vector<std::vector<double>> &A, 
    const std::vector<double> &b) {
    
    auto inv = gauss_jordan(A);
    int n = b.size();
    std::vector<double> x(n, 0.0);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            x[i] += inv[i][j] * b[j];
        }
    }
    
    return x;
}

std::vector<std::vector<double>> GaussJordan::gauss_jordan(const std::vector<std::vector<double>> &matrix) {
    size_t n = matrix.size();
    if (n == 0) {
        throw std::invalid_argument("matrix size is 0");
    }
    if (matrix[0].size() != n) {
        throw std::invalid_argument("Matrix must be square");
    }
    
    // extended matrix
    std::vector<std::vector<double>> result(n, std::vector<double>(2*n, 0.0));
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            result[i][j] = matrix[i][j];
        }
        result[i][i + n] = 1.0;
    }
    
    for (size_t i = 0; i < n; i++) {
        // find max elem
        double max_elem_in_row = std::abs(result[i][i]);
        size_t max_row = i;
        for (size_t k = i + 1; k < n; k++) {
            if (std::abs(result[k][i]) > max_elem_in_row) {
                max_elem_in_row = std::abs(result[k][i]);
                max_row = k;
            }
        }
        if (max_row != i) {
            for (size_t k = i; k < 2*n; k++) {
                std::swap(result[max_row][k], result[i][k]);
            }
        }
        
        if (std::abs(result[i][i]) < 1e-10) {
            throw std::runtime_error("degenerate matrix");
        }
        
        double pivot = result[i][i];
        for (int k = i; k < 2 * n; k++) {
            result[i][k] /= pivot;
        }
        for (int k = 0; k < n; k++) {
            if (k != i) {
                double factor = result[k][i];
                for (int j = i; j < 2 * n; j++) {
                    result[k][j] -= factor * result[i][j];
                }
            }
        }
    }
    
    std::vector<std::vector<double>> inverse(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inverse[i][j] = result[i][j + n];
        }
    }
    
    return inverse;
}
