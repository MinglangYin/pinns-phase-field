clear all
close all
clc

lmd = load('lmd_e.his');

figure()
plot(lmd)
hold on
yline(0)

loss = load('loss.his');
figure()
semilogy(loss)

min(loss)
lmd(find(loss == min(loss)))