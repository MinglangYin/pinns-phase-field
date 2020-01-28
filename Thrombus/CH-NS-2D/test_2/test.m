clear all
close all
clc

K = load('K.his');

figure()
plot(K)

loss = load('loss.his');
figure()
semilogy(loss)