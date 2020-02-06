clear all
close all
clc

% %% K
K = load('K.his');
figure()
semilogy(K)
hold on
yline(1)

%% loss
loss = load('loss.his');
figure()
semilogy(loss)

K(find(loss == min(loss)))
min_loss = min(loss)