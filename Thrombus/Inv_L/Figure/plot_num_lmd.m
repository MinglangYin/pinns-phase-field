clear all
close all
clc

%% figure 1: lmd_e_10
pts = [100, 10000, 20000, 30000];
pred_1 = [0.9783, 9.4829, 9.4829, 9.4260]; % velo + phase + psi
pred_2 = [9.8327, 3.2838, 3.13, 3.1001]; % velo + phase
pred_3 = [4.4390, 1.0536, 1.2464, 1.3618]; % phase

txt = '\downarrow real value';

subplot(1, 2, 1)
hold on
plot(pred_1, '*', 'Color', 'r', 'MarkerSize',8)
plot(pred_2, 'square', 'Color', 'b', 'MarkerSize',8);
plot(pred_3, 'o', 'Color', 'k', 'MarkerSize',8);
plot(pred_1, 'Color', 'r');
plot(pred_2, 'Color', 'b');
plot(pred_3, 'Color', 'k');
yline(10, '--','LineWidth',2)

xticklabels({'100','10000','20000', '30000'})
yticks([0 2.5 5 7.5 10 12.5 15])
set(gca, 'YGrid', 'on', 'XGrid', 'off')
xlim([1, 4])
ylim([0, 15]);

text(1, 10.4, txt)

legend('v+{\phi}+{\psi}+p', 'v+{\phi}', '{\phi}')
xlabel('Number of inner training points')
ylabel('{\lambda_{e}}')
title('case 1')

%% figure 2: lmd_e_10_K_001
pred_4 = [11.2741, 13.7730, 12.0356, 11.1850];% velo + phase + psi
pred_5 = [7.9832, 5.4431, 4.8335, 5.1053]; % velo + phase
pred_6 = [0.2300, 2.2079, 1.9436, 2.1127]; % velo

subplot(1, 2, 2)
hold on
plot(pred_4, '*', 'Color', 'r', 'MarkerSize',8)
plot(pred_5, 'square', 'Color', 'b', 'MarkerSize',8);
plot(pred_6, 'o', 'Color', 'k', 'MarkerSize',8);
plot(pred_4, 'Color', 'r');
plot(pred_5, 'Color', 'b');
plot(pred_6, 'Color', 'k');
yline(10, '--', 'LineWidth',2)

xticklabels({'100','10000','20000', '30000'})
yticks([0 2.5 5 7.5 10 12.5 15])
set(gca, 'YGrid', 'on', 'XGrid', 'off')
xlim([1, 4])
ylim([0, 15]);
text(1, 10.4, txt)

% legend('v+{\phi}+{\psi}+p', 'v+{\phi}', '{\phi}')
xlabel('Number of inner training points')
ylabel('{\lambda_{e}}')
title('case 2')