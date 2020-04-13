clear all
close all
clc

K = load('K.his');

figure()
semilogy(K)
hold on
yline(0.001)

loss = load('loss.his');
figure()
semilogy(loss)

K(find(loss == min(loss)))
min(loss)