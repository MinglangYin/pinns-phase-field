clear all
close all
clc

lmd = load('lmd_e.his');

figure()
plot(lmd)
hold on
yline(15)

loss = load('loss.his');
figure()
semilogy(loss(:,1))

min(loss)
lmd(end)