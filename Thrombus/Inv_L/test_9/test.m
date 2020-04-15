clear all
close all
clc

lmd = load('lmd_e.his');
lmd_1 = load('lmd_e_1.his');
lmd = [lmd_1', lmd'];
figure()
plot(lmd)
hold on
yline(10)
save('lmd.dat', 'lmd', '-ascii')

K = load('K.his');
K_1 = load('K_1.his');
K = [K_1', K'];
figure()
semilogy(K)
hold on
yline(0.01)
save('K.dat', 'K', '-ascii')

loss = load('loss.his');
loss_1 = load('loss_1.his');
loss = [loss_1', loss'];
figure()
semilogy(loss)
save('loss.dat', 'loss', '-ascii')

min(loss)
lmd(find(loss == min(loss)))