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

disp('K')
K(find(loss == min(loss)))
disp('loss')
min(loss)