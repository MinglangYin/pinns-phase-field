clear all
close all
clc

lmd = load('lmd_e.his');

figure()
plot(lmd)
hold on
yline(10)

K = load('K.his');
figure()
plot(K)
hold on
yline(0.01);

loss = load('loss.his');
figure()
semilogy(loss)

% min(loss)
lmd(end)
K(end)