clear all
close all
clc

% %% K
lmd_e = load('lmd_e.his');
figure()
plot(lmd_e)
hold on

%% loss
loss = load('loss.his');
figure()
semilogy(loss)

min_lmd_e = lmd_e(find(loss == min(loss)))
min_loss = min(loss)