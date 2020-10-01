clear all
close all
clc

K = load('K.his');

figure()
plot(K)
ylim([-5, 10])

figure()
x = linspace(-1, 0, 100);
y = exp(K(end, 1)*x) + K(end, 2);
plot(x, y)