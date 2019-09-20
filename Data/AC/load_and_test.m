clear all
close all
clc

data = load('AC.mat');

[X, T] = meshgrid(data.x, data.tt);

uu = data.uu;
surf(X, T, uu')