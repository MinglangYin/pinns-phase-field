clear all
close all
clc

lmd = load('lmd_e.his');

figure()
plot(lmd)
hold on
yline(10)

loss = load('loss.his');
figure()
semilogy(loss)

LOSS = sum(loss, 2);
min_loss = min(LOSS)
lmd(end)