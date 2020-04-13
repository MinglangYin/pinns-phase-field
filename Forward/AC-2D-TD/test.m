clear all
close all
clc
% 
% x = 0:0.01:1;
% y = 0:0.01:1;
% 
% [X, Y]=meshgrid(x, y);
% 
% f_init = @(x, y) tanh(         (0.1-sqrt((x-0.5).^2 + (y-0.5).^2))./sqrt(2)/0.01          );
% 
% surf(X, Y, f_init(X,Y))

% u_init = load('u_i.txt');
% u_init = reshape(u_init, [100, 100]);
% surf(u_init)

load('pred_res.mat')
% [X, Y]=meshgrid(x, y);
slice = squeeze(u(1,:,:));
surf(x, y, slice)