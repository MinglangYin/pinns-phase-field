clear all
close all
clc

K = load('K.his');

figure()
plot(K)
hold on
yline(100)

loss = load('loss.his');
figure()
semilogy(loss)

K(find(loss == min(loss)))