clear all
close all
clc

lmd_old = load('lmd_e_old.his');
lmd_new = load('lmd_e.his');
lmd = cat(1, lmd_old, lmd_new);

figure()
plot(lmd)
hold on
yline(10)

K_old = load('K_old.his');
K_new = load('K.his');
K = cat(1, K_old, K_new);

figure()
plot(K)
hold on
yline(0.01);

loss = load('loss.his');
loss_old = load('loss_old.his');
loss = cat(1, loss_old, loss);

figure()
semilogy(loss)

% min(loss)
lmd(end)
K(end)

save K.dat K -ascii
save lmd.dat lmd -ascii


