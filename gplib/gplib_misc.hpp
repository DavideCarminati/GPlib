#ifndef GPLIB_MISC
#define GPLIB_MISC

#include <Eigen/Core>

using namespace Eigen;

enum kernel_type_t
{
    GAUSSIAN,
    GAUSSIAN_ARD
};

enum device_t
{
    CPU,
    GPU
};

enum inference_type_t
{
    EXACT,
    FAST_APPROXIMATE
};

enum solver_type_t
{
    SR1,
    BFGS,
    ADAM
};

template <typename T>
Vector<T, -1> st_dev(const Matrix<T, -1, -1> &mat_in)
{
    const int p = mat_in.cols();
    Vector<T, -1> out(p);
    for (int idx = 0; idx < p; idx++)
    {
        out(idx) = sqrt( (mat_in.col(idx).array() - mat_in.col(idx).mean()).square().sum() / (mat_in.rows() - 1) );
    }
    return out;
};

template <typename T>
T st_dev(const Vector<T, -1> &vec_in)
{
    // const int p = vec_in.cols();
    T out;
    out = sqrt( (vec_in.array() - vec_in.mean()).square().sum() / (vec_in.rows() - 1) );
    return out;
};

#endif