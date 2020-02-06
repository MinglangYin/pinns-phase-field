clear all
close all
clc

% %% K
K = load('K.his');
figure()
semilogy(K)
hold on
yline(1)

%% lmd_e (wrong)
% lmd_e = load('lmd_e.his');
% figure()
% semilogy(lmd_e)
% hold on
% yline(0.5)

%% loss
loss = load('loss.his');
figure()
semilogy(loss)

K(find(loss == min(loss)))
lmd_e(find(loss == min(loss)))