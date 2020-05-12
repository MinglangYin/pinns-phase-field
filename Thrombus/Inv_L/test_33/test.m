clear all
close all
clc

lmd = load('lmd_e.his');
lmd_old = load('lmd_e_old.his');

figure()
plot(lmd)
hold on
yline(10)

loss = load('loss.his');
loss_old = load('loss_old.his');
figure()
semilogy(loss)

K = load('K.his');
K_old = load('K_old.his');
K_new = cat(1, K_old, K);
figure()
semilogy(K_new)

% min(loss)
lmd(end)

new_loss = cat(1, loss_old, loss);
new_lmd = cat(1, lmd_old, lmd);

save loss.dat new_loss -ascii
save lmd.dat new_lmd -ascii
save K.dat K_new -ascii

