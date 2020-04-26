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

loss_all = sum(loss, 2);
min(loss_all)
lmd(find(loss_all == min(loss_all)))