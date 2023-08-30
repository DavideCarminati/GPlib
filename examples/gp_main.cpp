#include <iostream>
#include <Eigen/Core>
#include <random>
#include <fstream>

#include "gplib.hpp"
// #include "gp.hpp"

#define GPLIB_USE_CUDA = 1

using namespace Eigen;

int main()
{
    // Building mathematical function
    std::random_device rd;
    std::mt19937 gen(rd());  
    std::normal_distribution<double> dis(0, 1e-3);

    int N = 10;
    int N_test = 20;
    VectorXd noise_vec = Eigen::VectorXd::Zero(N*N, 1).unaryExpr([&](double dummy){return dis(gen);});
    // VectorXd x_train = VectorXd::LinSpaced(N, -5, 5);
    MatrixXd X1_train = RowVectorXd::LinSpaced(N, -M_PI/2, M_PI/2).replicate(N, 1);
    MatrixXd X2_train = VectorXd::LinSpaced(N, -M_PI/2, M_PI/2).replicate(1, N);
    MatrixXd x_train(N*N, 2);
    x_train << X1_train.reshaped(N*N, 1), X2_train.reshaped(N*N, 1);
    // VectorXd x_test = VectorXd::LinSpaced(N_test, -5, 5);
    MatrixXd X1_test = RowVectorXd::LinSpaced(N_test, -M_PI/2, M_PI/2).replicate(N_test, 1);
    MatrixXd X2_test = VectorXd::LinSpaced(N_test, -M_PI/2, M_PI/2).replicate(1, N_test);
    MatrixXd x_test(N_test*N_test, 2);
    x_test << X1_test.reshaped(N_test*N_test, 1), X2_test.reshaped(N_test*N_test, 1);
    // VectorXd y_train_tmp = x_train.array() * sin(2 * x_train.array());
    VectorXd y_train_tmp = cos(x_train.col(0).array()) + cos(x_train.col(1).array());
    y_train_tmp += noise_vec;
    // VectorXd y_test_tmp = x_test.array() * sin(2 * x_test.array());
    VectorXd y_test_tmp = cos(x_test.col(0).array()) + cos(x_test.col(1).array());

    // Normalize
    VectorXd y_train = (y_train_tmp.array() - y_train_tmp.minCoeff()) / (y_train_tmp.maxCoeff() -  y_train_tmp.minCoeff());
    VectorXd y_test = (y_test_tmp.array() - y_test_tmp.minCoeff()) / (y_test_tmp.maxCoeff() - y_test_tmp.minCoeff());

    // gp_settings_t<double> user_setting(x_train, y_train);
    // user_setting.
    GaussianProcess<double> gp(x_train, y_train);
    gp.gp_settings.inference_type = FAST_APPROXIMATE;

    gp.train();

    // posterior_t<double> gp_posterior;
    // gp_posterior = gp.predict(x_test);
    // VectorXd y_star = gp_posterior.y_predicted;

    // // Writing on file
    // const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

    // std::ofstream file_X1_train("X1_train.csv");
    // file_X1_train << X1_train.format(CSVFormat);

    // std::ofstream file_X2_train("X2_train.csv");
    // file_X2_train << X2_train.format(CSVFormat);

    // std::ofstream file_y_train("y_train.csv");
    // file_y_train << y_train.format(CSVFormat);

    // std::ofstream file_X1_test("X1_test.csv");
    // file_X1_test << X1_test.format(CSVFormat);

    // std::ofstream file_X2_test("X2_test.csv");
    // file_X2_test << X2_test.format(CSVFormat);

    // std::ofstream file_y_test("y_test.csv");
    // file_y_test << y_test.format(CSVFormat);

    // std::ofstream file_y_star("y_predicted.csv");
    // file_y_star << y_star.format(CSVFormat);

    return 0;
}