clear all
close all
clc

lmd = load('lmd_e.his');

figure()
semilogy(lmd)
hold on
yline(0.25)

loss = load('loss.his');
figure()
semilogy(loss(:,1))

min(loss)
lmd(end)