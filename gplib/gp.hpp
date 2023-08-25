#ifndef GP
#define GP

#include <iostream>
#include <Eigen/Core>

// #include "gplib.hpp"
#include "kernel.hpp"
#include "gplib_misc.hpp"
#include "gplib_structs.hpp"

using namespace Eigen;

template <typename T>
class GaussianProcess
{
private:
    gp_settings_t<T> gp_settings;
    Kernel<T> kernel;
    MatrixXd X_train;
    VectorXd y_train;
    MatrixXf X_train_f;
    VectorXf y_train_f;
    Vector<T, -1> y_star;
    Matrix<T, -1, -1> cov_star;
    Matrix<T, -1, -1> inv_Ky, inv_K;    // Kernel inversions
    int N;                              // Number of train points
    int p;                              // Dimension of the problem
public:
    // GaussianProcess();
    // GaussianProcess(gp_settings_t<T> &user_gp_settings);
    GaussianProcess(Matrix<T, -1, -1> &X_train, Vector<T, -1> &y_train);
    GaussianProcess(Matrix<T, -1, -1> &X_train, Vector<T, -1> &y_train, gp_settings_t<T> &user_gp_settings);
    ~GaussianProcess();
    void train(void);
    posterior_t<T> predict(Matrix<T, -1, -1> &X_test);

    // Vector<T, -1> y_star;
    // Matrix<T, -1, -1> cov_star;
};


template <typename T>
GaussianProcess<T>::GaussianProcess(Matrix<T, -1, -1> &X_train, Vector<T, -1> &y_train) :   gp_settings(X_train, y_train)
{
    this->X_train.cast<T>();
    this->X_train = X_train;
    this->y_train.cast<T>();
    this->y_train = y_train;

    this->N = X_train.rows();
    this->p = X_train.cols();
}

template <typename T>
GaussianProcess<T>::GaussianProcess(Matrix<T, -1, -1> &X_train, Vector<T, -1> &y_train, gp_settings_t<T> &user_gp_settings)
{
    this->X_train.cast<T>();
    this->X_train = X_train;
    this->y_train.cast<T>();
    this->y_train = y_train;

    this->gp_settings = user_gp_settings;
    this->N = X_train.rows();
    this->p = X_train.cols();
}

template <typename T>
GaussianProcess<T>::~GaussianProcess()
{
    //
}

template <typename T>
void GaussianProcess<T>::train()
{
    // Check if exact inference or not
    switch (this->gp_settings.inference_type)
    {
    case EXACT:
        {
            Matrix<T, -1, -1> K = this->kernel.computeK(this->X_train, this->gp_settings.hyperparameters, this->gp_settings.kernel_type);
            // Compute K^-1 * y_train
            inv_Ky = K.lu().solve(this->y_train);
            // Compute K^-1
            this->inv_K = K.lu().solve(Matrix<T, -1, -1>::Identity(this->N, this->N));
            
            break;
        }
    case FAST_APPROXIMATE:
        //
        break;

    default:
        throw "Inference type not available.";
        break;
    }
    
}

template <typename T>
posterior_t<T> GaussianProcess<T>::predict(Matrix<T, -1, -1> &X_test)
{
    switch (this->gp_settings.inference_type)
    {
    case EXACT:
        {
            Matrix<T, -1, -1> Kstar = this->kernel.computeKstar(X_test);
            Matrix<T, -1, -1> Kstarstar = this->kernel.computeKstarstar(X_test);
            this->y_star = Kstar * this->inv_Ky;
            this->cov_star = Kstarstar - Kstar * this->inv_K * Kstar.transpose();
            break;
        }
    
    default:
        break;
    }
    posterior_t<T> gp_posterior;
    gp_posterior.y_predicted = this->y_star;
    gp_posterior.y_covariance = this->cov_star;
    return gp_posterior;
}

#endif