clear all
close all
clc

lmd_new = load('lmd_e.his');
lmd_old = load('lmd_old.his');
lmd = cat(1, lmd_old,lmd_new);

figure()
semilogy(lmd)
hold on
yline(0.25)

loss_old = load('loss_old.his');
loss_new = load('loss.his');
loss = cat(1, loss_old, loss_new);

figure()
semilogy(loss(:,1))

min(loss)
lmd(end)

save('lmd.dat', 'lmd', '-ascii');
save('loss.dat', 'loss', '-ascii');