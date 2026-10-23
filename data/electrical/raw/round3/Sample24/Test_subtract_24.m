%% PUND switching current subtraction

clear;
clc;
close all;

%% Load data
% Example:
pund = readmatrix('PUND_pris_1e6_R9C9_30um.xls','Sheet','Data','Range','B2:D3401');

t = pund(:,3);
I = pund(:,2);
V = pund(:,1);

%% Define corresponding indices

I_switch_idx    = 1800:2000;
I_nonswitch_idx = 2600:2800;

%% Extract

t_switch = t(I_switch_idx);
I_switch = I(I_switch_idx);

t_nonswitch = t(I_nonswitch_idx);
I_nonswitch = I(I_nonswitch_idx);

%% Check that lengths match

if length(I_switch) ~= length(I_nonswitch)
    error('Switching and non-switching index ranges must have the same length.');
end

%% Subtract

I_subtracted = I_switch - I_nonswitch;

%% Plot

figure;
plot(t_switch, I_switch, 'k-', 'LineWidth', 1.2);
hold on;
plot(t_switch, I_nonswitch, 'b--', 'LineWidth', 1.2);
plot(t_switch, I_subtracted, 'r-', 'LineWidth', 2);

xlabel('Time');
ylabel('Current');
legend('Switching','Non-switching','Subtracted');
grid on;
box on;

title('PUND Switching Current');

%% 
% Smooth the subtracted current first
I_smooth = sgolayfilt(I_switch, 3, 21);

% Take derivative
dIdt = gradient(I_smooth, t_switch);

% Plot derivative
figure;
plot(t_switch, dIdt, 'k-', 'LineWidth', 1.5);
hold on;
yline(0, 'r--', 'LineWidth', 1.2);

xlabel('Time');
ylabel('dI/dt');
title('Derivative of Subtracted Switching Current');
grid on;
box on;

yyaxis right
plot(t_switch, I_smooth, 'r-', 'LineWidth', 1.2);
ylabel('Current (A)')