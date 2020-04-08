clear all
close all
clc

K = load('K.his');

figure()
semilogy(K)
hold on
yline(0.01)

loss = load('loss.his');
figure()
semilogy(loss)

min(loss)
K(find(loss == min(loss)))