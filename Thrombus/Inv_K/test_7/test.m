clear all
close all
clc

K = load('K.his');
K_old = load('K_old.his');
K_new = cat(1, K_old, K);
save('K_new.his', 'K_new','-ascii')

figure()
plot(K_new)
hold on
yline(100)

loss = load('loss.his');
figure()
semilogy(loss)

K(find(loss == min(loss)))