#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>

// #include "gplib.hpp"
#include "gplib_misc.hpp"
#include "gplib_structs.hpp"

using namespace Eigen;

template <typename T>
class Kernel
{
private:
    Matrix<T, -1, -1> X_train;//, X_test;
    // Vector<T, -1> y_train, y_test;
    kernel_type_t kernel_type;
    hyperparameters_t<T> hyperparameters;

    void gaussian(  const Matrix<T, -1, -1> &x1,
                    const Matrix<T, -1, -1> &x2, 
                    const T amplitude, 
                    const T length_scale,
                    Matrix<T, -1, -1> &kernel
                    );

    void gaussian_ard(  const Matrix<T, -1, -1> &x1,
                        const Matrix<T, -1, -1> &x2, 
                        const T amplitude, 
                        const Vector<T, -1> &length_scale,
                        Matrix<T, -1, -1> &kernel
                        );
public:
    Kernel();
    ~Kernel();
    Matrix<T, -1, -1> computeK(Matrix<T, -1, -1> &X_train, hyperparameters_t<T> &hyp, kernel_type_t &type);
    Matrix<T, -1, -1> computeKstar(const Matrix<T, -1, -1> &X_test);
    Matrix<T, -1, -1> computeKstarstar(const Matrix<T, -1, -1> &X_test);

    Matrix<T, -1, -1> K, K_star, K_starstar;
};



template <typename T>
Kernel<T>::Kernel()
{
    //
}


template <typename T>
Matrix<T, -1, -1> Kernel<T>::computeK(Matrix<T, -1, -1> &X_train, hyperparameters_t<T> &hyp, kernel_type_t &kernel_type)
{
    this->K.resize(X_train.rows(), X_train.rows());
    this->X_train = X_train;
    this->kernel_type = kernel_type;
    this->hyperparameters = hyp;

    switch (kernel_type)
    {
    case GAUSSIAN:
        this->gaussian(X_train, X_train, hyp.amplitude, hyp.length_scale(0), this->K);
        break;
    
    case GAUSSIAN_ARD:
        this->gaussian_ard(X_train, X_train, hyp.amplitude, hyp.length_scale, this->K);
        break;

    default:
        break;
    }
    return this->K;
}

template <typename T>
Kernel<T>::~Kernel()
{
}

template <typename T>
Matrix<T, -1, -1> Kernel<T>::computeKstar(const Matrix<T, -1, -1> &X_test)
{
    this->K_star.resize(this->K.rows(), this->K_star.rows());
    switch (this->kernel_type)
    {
    case GAUSSIAN:
        this->gaussian(X_test, X_train, this->hyperparameters.amplitude, this->hyperparameters.length_scale(0), this->K_star);
        break;
    
    case GAUSSIAN_ARD:
        this->gaussian_ard(X_test, X_train, this->hyperparameters.amplitude, this->hyperparameters.length_scale, this->K_star);
        break;

    default:
        break;
    }
    return this->K_star;
}

template <typename T>
Matrix<T, -1, -1> Kernel<T>::computeKstarstar(const Matrix<T, -1, -1> &X_test)
{
    this->K_starstar.resize(this->K.rows(), this->K_star.rows());
    switch (this->kernel_type)
    {
    case GAUSSIAN:
        this->gaussian(X_test, X_test, this->hyperparameters.amplitude, this->hyperparameters.length_scale(0), this->K_starstar);
        break;
    
    case GAUSSIAN_ARD:
        this->gaussian_ard(X_test, X_test, this->hyperparameters.amplitude, this->hyperparameters.length_scale, this->K_starstar);
        break;

    default:
        break;
    }
    return this->K_starstar;
}

/**
 * Gaussian kernel
*/
template <typename T>
void Kernel<T>::gaussian(  const Matrix<T, -1, -1> &x1,
                        const Matrix<T, -1, -1> &x2, 
                        const T amplitude, 
                        const T length_scale,
                        Matrix<T, -1, -1> &kernel
                        )
{
    // Checking input size
    if (x1.cols() != x2.cols()) 
    {
        throw std::invalid_argument("x1 and x2 must have the same number of columns");
    }

    Array<T, -1, 1> l;
    l.resize(x1.cols(), Eigen::NoChange);
    l.setConstant(length_scale);
    Matrix<T, -1, -1> Lambda = (1 / l.square()).matrix().asDiagonal();

    // Matrix<T, -1, -1> kernel(x1.rows(), x2.rows()); // Initialize kernel matrix
    kernel.resize(x1.rows(), x2.rows());
    for (int ii = 0; ii < x1.rows(); ii++)
    {
        for (int jj = 0; jj < x2.rows(); jj++)
        {
            kernel(ii,jj) = amplitude * amplitude * exp( -0.5 * (x1.row(ii) - x2.row(jj)) * Lambda * (x1.row(ii) - x2.row(jj)).transpose() );
        }
    }
    return;
}

/**
 * Gaussian kernel with Automatic relevance Determination (ARD)
*/
template <typename T>
void Kernel<T>::gaussian_ard(  const Matrix<T, -1, -1> &x1,
                            const Matrix<T, -1, -1> &x2, 
                            const T amplitude, 
                            const Vector<T, -1> &length_scale,
                            Matrix<T, -1, -1> &kernel
                            )
{
    // Checking input size
    if (x1.cols() != x2.cols()) 
    {
        throw std::invalid_argument("x1 and x2 must have the same number of columns");
    }

    Array<T, -1, 1> l = length_scale.array();
    Matrix<T, -1, -1> Lambda = (1 / l.square()).matrix().asDiagonal();

    // Matrix<T, -1, -1> kernel(x1.rows(), x2.rows()); // Initialize kernel matrix
    kernel.resize(x1.rows(), x2.rows());
    for (int ii = 0; ii < x1.rows(); ii++)
    {
        for (int jj = 0; jj < x2.rows(); jj++)
        {
            kernel(ii,jj) = amplitude * amplitude * exp( -0.5 * (x1.row(ii) - x2.row(jj)) * Lambda * (x1.row(ii) - x2.row(jj)).transpose() );
        }
    }
    return;
}


