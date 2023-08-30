#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <random>
#include <assert.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <boost/math/special_functions/hermite.hpp>
#include <cublas_v2.h>
#include <chrono>
#include <fstream>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/LU>
// #include <cuLinearSolver.hpp>
// #include "cuFAGPutils.hpp"
// #include "cusolver_utils.hpp"



// #define N 100//0000
#define MAX_ERR 1e-6

using namespace Eigen;


template<typename M>
M load_csv (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}

MatrixXi all_combinations(const int n, const int p)
{
    MatrixXi all_comb((int)pow(n, p), p);
    std::vector<MatrixXi> grids;
    MatrixXi index1 = VectorXi::LinSpaced(n, 1, n).replicate(1, (int)pow(n, p-1));
    grids.push_back(index1);
    for (int ii = 1; ii < p; ii++)
    {
        MatrixXi index_ii = RowVectorXi::LinSpaced(n, 1, n).replicate((int)pow(n, ii), (int)pow(n, p - ii - 1));
        grids.push_back(index_ii);
        // grids[ii] = index_ii;
    }
    int idx = 0;
    for (auto grid : grids)
    {
        all_comb.col(idx++) = grid.reshaped((int)pow(n, p), 1);
    }
    return all_comb;

}

template <typename T>
int cuTrain(const Matrix<T, -1, -1> &x_train, hyperparameters_t<T> &hyp)
{
    // MatrixXd x_train = load_csv<MatrixXd>("../input_matrices/x_train.csv");
    // MatrixXi eig_comb = eig_comb_in.cast<int>();

    MatrixXi eig_comb = all_combinations(5,4);
    std::cout << "EIg_comb is a " << eig_comb.rows() << "x" << eig_comb.cols() << " and contains:\n" << eig_comb.topRows(11) << std::endl;
    // TODO create a hyperparameter struct specifically for fast approximate GP!!
    // TODO add a class template to gp_settings struct for the hyperparameters used. 

    /*
    // Normalize
    VectorXd y_train = (y_train_tmp.array() - y_train_tmp.minCoeff()) / (y_train_tmp.maxCoeff() -  y_train_tmp.minCoeff());
    VectorXd y_test = (y_test_tmp.array() - y_test_tmp.minCoeff()) / (y_test_tmp.maxCoeff() - y_test_tmp.minCoeff());

    const int N = y_train.rows();
    const int N_test = y_test.rows();
    const int p = x_train.cols();       // Number of dimensions of the problem
    std::cout << "# of training points: " << N << "\n# of test points: " << N_test << "\n# of problem dimension: " << p << 
             "\n# of eigenvalues: " << pow(eig_comb.rows(), 1/(double)p) << std::endl;

    // std::cout << "x_train is\n" << x_train.head(10) << std::endl;

    // GP parameters
    double l = 1;
    double epsilon = 1 / (sqrt(2) * l);
    double alpha = 0.5;
    int n = 8; // Number of eigenvaluessss
    int np = pow(n, p);     // Number of total n^p combination of eigenvalues
    double sigma_n = 1/pow(1e-3, 2);
    double minus_sigma_n2 = -1 / pow(1e-3, 4);
    MatrixXd identity = MatrixXd::Identity(N, N);

    // // Create matrix of eigenvalues combinations
    // MatrixXi index1 = VectorXi::LinSpaced(n, 1, n).replicate(1, n);
    // MatrixXi index2 = RowVectorXi::LinSpaced(n, 1, n).replicate(n, 1);
    // // std::cout << "size index1 " << index1.rows() << "x" << index1.cols() << std::endl;
    // // std::cout << "index1 " << index1 << std::endl;
    // // std::cout << "size index2 " << index2.rows() << "x" << index2.cols() << std::endl;
    // // std::cout << "index2 " << index2 << std::endl;
    // MatrixXi eig_comb(np, 2);
    // eig_comb << index1.reshaped(np, 1), index2.reshaped(np, 1);



    auto gpu_start = std::chrono::steady_clock::now();

    // Allocate device memory
    double *dev_x_train, *dev_x_test, *dev_y_train;                                         // Datasets
    double *dev_l, *dev_epsilon, *dev_alpha, *dev_sigma_n, *dev_minus_sigma_n;              // Hyperparameters
    double *dev_Phi, *dev_Phip, *dev_Lambda, *dev_inv_Lambda, *dev_identity, *dev_Phi_T, *dev_lambdas;
    int *dev_eig_comb;
    CUDA_CHECK(cudaMalloc((void**)&dev_x_train, sizeof(double) * x_train.size()));
    CUDA_CHECK(cudaMalloc((void**)&dev_x_test, sizeof(double) * x_test.size()));
    CUDA_CHECK(cudaMalloc((void**)&dev_y_train, sizeof(double) * N));

    CUDA_CHECK(cudaMalloc((void**)&dev_l, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&dev_epsilon, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&dev_alpha, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&dev_sigma_n, sizeof(sigma_n)));
    CUDA_CHECK(cudaMalloc((void**)&dev_minus_sigma_n, sizeof(sigma_n)));

    CUDA_CHECK(cudaMalloc((void**)&dev_eig_comb, sizeof(int) * np * p));
    CUDA_CHECK(cudaMalloc((void**)&dev_Phi, sizeof(double) * N * pow(n, p)));
    CUDA_CHECK(cudaMalloc((void**)&dev_Phip, sizeof(double) * N_test * np));
    CUDA_CHECK(cudaMalloc((void**)&dev_Lambda, sizeof(double) * np * np ));
    CUDA_CHECK(cudaMalloc((void**)&dev_inv_Lambda, sizeof(double) * np * np ));
    CUDA_CHECK(cudaMalloc((void**)&dev_identity, sizeof(double) * identity.size()));
    CUDA_CHECK(cudaMalloc((void**)&dev_Phi_T, sizeof(double) * N * pow(n, p)));
    CUDA_CHECK(cudaMalloc((void**)&dev_lambdas, sizeof(double) * np));
    // Results
    double *dev_y_star;
    CUDA_CHECK(cudaMalloc((void**)&dev_y_star, sizeof(double) * N_test));

    // Allocating temporary variables for the computation of W
    double *dev_tmp_W1, *dev_tmp_W2;
    CUDA_CHECK(cudaMalloc((void**)&dev_tmp_W1, sizeof(double) * N * np));
    CUDA_CHECK(cudaMalloc((void**)&dev_tmp_W2, sizeof(double) * N * N));
    // Allocating temporary variables for the computation of Phip * Lambda * Phi
    double *dev_tmp_Kstar, *dev_Kstar;
    CUDA_CHECK(cudaMalloc((void**)&dev_tmp_Kstar, sizeof(double) * N_test * np));
    CUDA_CHECK(cudaMalloc((void**)&dev_Kstar, sizeof(double) * N_test * N));
    double *dev_W;
    CUDA_CHECK(cudaMalloc((void**)&dev_W, sizeof(double) * N_test * N));

    CUDA_CHECK(cudaMemcpy(dev_x_train, x_train.data(), sizeof(double) * x_train.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_x_test, x_test.data(), sizeof(double) * x_test.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_y_train, y_train.data(), sizeof(double) * y_train.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_sigma_n, &sigma_n, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_eig_comb, eig_comb.data(), sizeof(int) * eig_comb.size(), cudaMemcpyHostToDevice));
    // cudaMemcpy(dev_minus_sigma_n, &minus_sigma_n, sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(dev_Phi, Phi.data(), sizeof(double) * Phi.size(), cudaMemcpyHostToDevice);
    // cudaMemcpy(dev_Phip, Phip.data(), sizeof(double) * Phip.size(), cudaMemcpyHostToDevice);
    // cudaMemcpy(dev_Lambda, Lambda.data(), sizeof(double) * Lambda.size(), cudaMemcpyHostToDevice);
    // cudaMemcpy(dev_inv_Lambda, inv_Lambda.data(), sizeof(double) * inv_Lambda.size(), cudaMemcpyHostToDevice);
    CUDA_CHECK(cudaMemcpy(dev_identity, identity.data(), sizeof(double) * identity.size(), cudaMemcpyHostToDevice));
    
    // cudaMemcpy(dev_Phi_T, Phi_T.data(), sizeof(double) * Phi_T.size(), cudaMemcpyHostToDevice);

    CUDA_CHECK(cudaMemcpy(dev_tmp_W2, identity.data(), sizeof(double) * N * N, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    // cudaStream_t stream = NULL;
    // cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cublasCreate(&handle);

    // cudaMemcpyAsync(dev_sigma_n, &sigma_n, sizeof(double), cudaMemcpyHostToDevice, stream);

    const double one = 1.0, zero = 0.0;

    // Creating eigenvector matrices Phi, Phip. Creating eigenvalues
    int block_size = 256;
    int grid_size = ((N + block_size) / block_size);

    // Inference
    eigenFunction<<<grid_size, block_size>>>(dev_x_train, N, p, dev_eig_comb, np, epsilon, alpha, dev_Phi, dev_Phi_T);
    eigenFunction<<<grid_size, block_size>>>(dev_x_test, N_test, p, dev_eig_comb, np, epsilon, alpha, dev_Phip, nullptr);
    eigenValues<<<grid_size, block_size>>>(dev_eig_comb, np, p, epsilon, alpha, dev_Lambda, dev_inv_Lambda);

    // dev_inv_Lambda becomes Lambda_hat
    // std::cout << "Phi is " << Phi.rows() << "x" << Phi.cols() << std::endl;
    // std::cout << "Inv_Lambda is " << inv_Lambda.rows() << "x" << inv_Lambda.cols() << std::endl;
    // Note that the leading dimensions lda, ldb and ldc do not change if I use the transpose in the computation! 
    // In Eigen, since is Column-major, the lead dim is the # of rows
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, np, np, N,
                    &sigma_n, dev_Phi, N, dev_Phi, N, &one, dev_inv_Lambda, np));

    // Checking dev_Phi_T
    // MatrixXd check(n, N);
    // cudaMemcpy(check.data(), dev_Phi_T, sizeof(double) * n * N, cudaMemcpyDeviceToHost);
    // std::cout << "Phi_T is a " << check.rows() << "x" << check.cols() << " matrix. Contains:\n" << check << std::endl;

    // Comparison between Lambda_hat su Cpu e gpu

    // std::cout << "Host Lambda_hat =\n " << h_Lambda_hat << std::endl;

    // Computing Lambda_hat^-1 * Phi.transpose(). Solution is stored in dev_Phi_T
    cuLinearSolver(dev_inv_Lambda, np, dev_Phi_T, N);

    // Finding W.
    // // Solution is stored in dev_tmp_W1. dev_tmp_W1 = sigma_n * Phi * (Lambda_hat)^-1
    // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Phi.rows(), n, Phi.cols(), dev_sigma_n, dev_Phi, Phi.rows(), dev_identity, n, 0, dev_tmp_W1, N);
    // // Solution is stored is dev_tmp_W2. dev_tmp_W2 = dev_tmp_W1 * Phi.transpose() * sigma_n
    // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, N, n, dev_minus_sigma_n, dev_tmp_W1, N, dev_Phi, N, dev_sigma_n, dev_tmp_W2, N);

    // Computing Sigma_n^-1 - sigma_n^2 * Phi * (Lambda_hat^-1 * Phi.transpose()). Lambda_hat^-1 * Phi.transpose() was computed using
    // cuLinearSolver(). Solution is stored in dev_tmp_W2 (initialized as identity matrix to pass Sigma_n^-1)
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, np, &minus_sigma_n2, dev_Phi, N, dev_Phi_T, np, &sigma_n, dev_tmp_W2, N));

    
    // Finding Phip * Lambda * Phi
    // double* one;
    // *one = 1.0;
    
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N_test, np, np, &one, dev_Phip, N_test, dev_Lambda, np, &zero, dev_tmp_Kstar, N_test));

    MatrixXd Kstar_tmp(N_test, np);
    CUDA_CHECK(cudaMemcpy(Kstar_tmp.data(), dev_tmp_Kstar, sizeof(double) * np * N_test, cudaMemcpyDeviceToHost));
    // std::cout << "Kstar_tmp is a " << Kstar_tmp.rows() << "x" << Kstar_tmp.cols() << " matrix. Contains:\n" << Kstar_tmp << std::endl;

    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N_test, N, np, &one, dev_tmp_Kstar, N_test, dev_Phi, N, &zero, dev_Kstar, N_test));

    MatrixXd Kstar(N_test, N);
    CUDA_CHECK(cudaMemcpy(Kstar.data(), dev_Kstar, sizeof(double) * N * N_test, cudaMemcpyDeviceToHost));
    // std::cout << "Kstar is a " << Kstar.rows() << "x" << Kstar.cols() << " matrix. Contains:\n" << Kstar << std::endl;

    // Computing W
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N_test, N_test, N, &one, dev_Kstar, N_test, dev_tmp_W2, N, &zero, dev_W, N_test));

    MatrixXd W(N_test, N);
    CUDA_CHECK(cudaMemcpy(W.data(), dev_W, sizeof(double) * N * N_test, cudaMemcpyDeviceToHost));
    // std::cout << "W is a " << W.rows() << "x" << W.cols() << " matrix. Contains:\n" << W << std::endl;

    // Computing posterior mean y_star and covariance cov_star
    const int incx = 1;
    const int incy = 1;
    CUBLAS_CHECK(cublasDgemv(handle, CUBLAS_OP_N, N_test, N, &one, dev_W, N_test, dev_y_train, incx, &zero, dev_y_star, incy));

    // cudaStreamDestroy(stream);

    VectorXd y_star(N_test);
    CUDA_CHECK(cudaMemcpy(y_star.data(), dev_y_star, sizeof(double) * N_test, cudaMemcpyDeviceToHost));

    cudaDeviceReset();

    auto elapsed_gpu = (std::chrono::steady_clock::now() - gpu_start);
    
    // std::cout << "CPU took " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_cpu).count() << "ms." << std::endl;
    std::cout << "GPU took " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_gpu).count() << "ms." << std::endl;

    // std::cout << "Posterior mean is\n" << y_star << std::endl;

    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

    std::ofstream file_x_train("../output_matrices/x_train.csv");
    file_x_train << x_train.format(CSVFormat);

    // std::ofstream file_X2_train("X2_train.csv");
    // file_X2_train << X2_train.format(CSVFormat);

    std::ofstream file_y_train("../output_matrices/y_train.csv");
    file_y_train << y_train.format(CSVFormat);

    std::ofstream file_x_test("../output_matrices/x_test.csv");
    file_x_test << x_test.format(CSVFormat);

    // std::ofstream file_X2_test("X2_test.csv");
    // file_X2_test << X2_test.format(CSVFormat);

    std::ofstream file_y_test("../output_matrices/y_test.csv");
    file_y_test << y_test.format(CSVFormat);

    std::ofstream file_y_star("../output_matrices/y_predicted.csv");
    file_y_star << y_star.format(CSVFormat);*/

}