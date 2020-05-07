clear all
close all
clc

lmd = load('lmd_e.his');

figure()
plot(lmd(1:300000))
hold on
yline(10)

loss = load('loss.his');
figure()
semilogy(loss(1:300000))

% min(loss)
lmd(300000)