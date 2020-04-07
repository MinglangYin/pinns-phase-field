clear all
close all
clc

K = load('K.his');

figure()
plot(K)
hold on
yline(1)

loss = load('loss.his');
figure()
semilogy(loss)