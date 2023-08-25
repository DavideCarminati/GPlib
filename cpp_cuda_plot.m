%% FAGP with cuda
% The script reads and plots the result of CUDA FAGP saved in csv format.

close all
clear
clc

load("build/X1_train.csv");
load("build/X2_train.csv");
load("build/y_train.csv");

load("build/X1_test.csv");
load("build/X2_test.csv");
load("build/y_test.csv");
load("build/y_predicted.csv");

figure
subplot(1, 2, 1)
surf(X1_test, X2_test, reshape(y_test, size(X1_test, 1), size(X1_test, 1)));
grid on;
subplot(1, 2, 2)
surf(X1_train, X2_train, reshape(y_train, size(X1_train, 1), size(X1_train, 1)));
grid on;

figure
hold on
surf(X1_test, X2_test, reshape(y_test, size(X1_test, 1), size(X1_test, 1)), 'EdgeColor','none', 'FaceAlpha', 0.5);
grid on;
surf(X1_test, X2_test, reshape(y_predicted, size(X1_test, 1), size(X1_test, 1)));
% colormap gray