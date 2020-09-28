clear all
close all
clc

K = load('K.his');
K_old = load('K_1.his');
% K_new = cat(1, K_old, K);
% save('K_new.his', 'K_new','-ascii')

figure();
semilogy(K);
hold on;
semilogy(K_old);
yline(1)

loss = load('loss.his');
loss_old = load('loss_1.his');
figure();
semilogy(loss);
hold on;
semilogy(loss_old);

% K(find(loss == min(loss)))

