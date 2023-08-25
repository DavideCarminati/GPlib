#ifndef GPLIB_STRUCTS
#define GPLIB_STRUCTS

#include <string>
#include <Eigen/Core>

#include "gplib_misc.hpp"

using namespace Eigen;

template <typename T>
struct posterior_t
{
    Vector<T, -1> y_predicted;
    Matrix<T, -1, -1> y_covariance;
    // Add uncertainty intervals
};

template <typename T = double>
struct hyperparameters_t
{
    int p;
    T amplitude;
    Vector<T, -1> length_scale;
    T noise_st_dev;
    void set_train_data(Matrix<T, -1, -1> x_samples, Vector<T, -1> &y_samples)// : X_train(x_samples), y_train(y_samples)
    {
        p = x_samples.cols();     // Problem dimension
        amplitude = 1.0;
        length_scale.resize(p);
        length_scale = st_dev<T>(x_samples);
        noise_st_dev = st_dev<T>(y_samples);    
    }
    // private:
    //     Matrix<T, -1, -1> X_train;
    //     Vector<T, -1> y_train;
};

template <typename T = double>
struct gp_settings_t
{
    gp_settings_t(Matrix<T, -1, -1> x_samples, Vector<T, -1> &y_samples)// : hyperparameters(x_samples, y_samples)
    {
        // hyperparameters_t<T> hyperparameters(X_train, y_train);
        hyperparameters.set_train_data(x_samples, y_samples);
    };
    hyperparameters_t<T> hyperparameters;
    inference_type_t inference_type = EXACT;
    kernel_type_t kernel_type = GAUSSIAN;
    device_t device = CPU;
    bool optimize_hyperparameters = true;
    bool estimate_hyperparameters = false;
    solver_type_t solver_type = SR1;
    private:
        Matrix<T, -1, -1> X_train;
        Vector<T, -1> y_train;
};



#endif