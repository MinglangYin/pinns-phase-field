clear all
close all
clc

load("pred_res.mat")

% u_grid = reshape(u, [512, 201]);
surf(u)
% 
% x_grid = reshape(x, [100, 50]);
% t_grid = reshape(t, [100, 50]);
% u_grid = reshape(u, [100, 50]);
% 
% surf(x_grid, t_grid, u_grid)
