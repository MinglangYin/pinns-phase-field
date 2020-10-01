clear all
close all
clc

K = load('K.his');

figure()
plot(K)
ylim([-5, 10])
legend()

figure()
x = linspace(-1, 1, 100);
% y = K(end, 1)*tanh(K(end, 2)*x + K(end, 3)) + K(end, 4);
y = K(end, 1)*tanh(5*x + K(end, 3)) + K(end, 4);
plot(x, y)