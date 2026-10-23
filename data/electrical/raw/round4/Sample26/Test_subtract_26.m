%% PUND switching current subtraction

clear;
clc;
close all;

%% Load data
% Example:
pund = readmatrix('PUND_pris_1e6_R8C8_30um.xls','Sheet','Data','Range','B2:D3401');

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

% %% Subtract

% I_subtracted = I_switch - I_nonswitch;

% %% Plot
% 
% figure;
% plot(t_switch, I_switch, 'k-', 'LineWidth', 1.2);
% hold on;
% plot(t_switch, I_nonswitch, 'b--', 'LineWidth', 1.2);
% plot(t_switch, I_subtracted, 'r-', 'LineWidth', 2);
% 
% xlabel('Time');
% ylabel('Current');
% legend('Switching','Non-switching','Subtracted');
% grid on;
% box on;
% 
% title('PUND Switching Current');

% %% 
% % Smooth the subtracted current first
% I_smooth = sgolayfilt(I_switch, 3, 21);
% 
% % Take derivative
% dIdt = gradient(I_smooth, t_switch);
% 
% % Plot derivative
% figure;
% plot(t_switch, dIdt, 'k-', 'LineWidth', 1.5);
% hold on;
% yline(0, 'r--', 'LineWidth', 1.2);
% 
% xlabel('Time');
% ylabel('dI/dt');
% title('Derivative of Subtracted Switching Current');
% grid on;
% box on;
% 
% yyaxis right
% plot(t_switch, I_smooth, 'r-', 'LineWidth', 1.2);
% ylabel('Current (A)')
% 
% %find peaks 
% [pks, locs] = findpeaks(I_smooth);
% % plot(t_switch(locs), pks, 'ro', 'MarkerFaceColor', 'r');
% xline(t_switch(locs))

%% Integrate areas - Switching area ratio and fractions

% Smooth raw switching current
I_smooth = sgolayfilt(I_switch, 3, 21);

% Derivative of smoothed current
dIdt = gradient(I_smooth, t_switch);

% Find POSITIVE -> NEGATIVE derivative zero crossings
% These correspond to local maxima in the current
zeroIdx = find(dIdt(1:end-1) > 0 & dIdt(2:end) <= 0);

% Interpolate derivative crossover times
tCross = zeros(size(zeroIdx));

for k = 1:length(zeroIdx)

    i = zeroIdx(k);

    tCross(k) = t_switch(i) - ...
        dIdt(i) * (t_switch(i+1) - t_switch(i)) / ...
        (dIdt(i+1) - dIdt(i));

end

% Find local maxima independently
peakProminence = 0.05 * (max(I_smooth) - min(I_smooth));

[pks, locs] = findpeaks(I_smooth, ...
    'MinPeakProminence', peakProminence);

tPeaks = t_switch(locs);

% Print detected crossover points
fprintf('\nDetected derivative crossovers:\n');

if isempty(tCross)
    fprintf('None detected.\n');
else
    fprintf('%.4f us\n', tCross*1e6);
end

% Print detected local maxima
fprintf('\nDetected local maxima:\n');

if isempty(tPeaks)
    fprintf('None detected.\n');
else
    fprintf('%.4f us\n', tPeaks*1e6);
end

% Find crossover closest to any local maximum
if isempty(tCross) || isempty(tPeaks)

    warning('Could not identify derivative crossovers and/or local maxima.');

else

    % Distance between every crossover and every local maximum
    distanceMatrix = abs(tCross(:) - tPeaks(:)');

    % Find smallest distance
    [minDistance, linearIdx] = min(distanceMatrix(:));

    [crossIdx, peakIdx] = ind2sub(size(distanceMatrix), linearIdx);

    % Selected separation point
    separationTime = tCross(crossIdx);

    % Find corresponding raw-data index
    separationIdx = find(t_switch >= separationTime, 1);

    % Integrate RAW switching current
    A1 = abs(trapz(t_switch(1:separationIdx), ...
                   I_switch(1:separationIdx)));

    A2 = abs(trapz(t_switch(separationIdx:end), ...
                   I_switch(separationIdx:end)));

    % Area ratio
    ratio = A1 / A2;

    % Normalized fractions
    fraction1 = A1 / (A1 + A2);
    fraction2 = A2 / (A1 + A2);

    % Print results
    fprintf('\nSelected crossover = %.4f us\n', ...
        separationTime*1e6);

    fprintf('Matched local maximum = %.4f us\n', ...
        tPeaks(peakIdx)*1e6);

    fprintf('Crossover-to-maximum difference = %.4f us\n', ...
        minDistance*1e6);

    fprintf('\nArea 1 = %.4e A*s\n', A1);
    fprintf('Area 2 = %.4e A*s\n', A2);

    fprintf('\nArea ratio A1/A2 = %.4f\n', ratio);

    fprintf('\nFraction 1 = %.2f %%\n', fraction1*100);
    fprintf('Fraction 2 = %.2f %%\n', fraction2*100);

    % Plot
    figure;
    hold on;

    % First integrated area
    t1 = t_switch(1:separationIdx);
    I1 = I_switch(1:separationIdx);

    fill([t1; flipud(t1)], ...
         [I1; zeros(size(I1))], ...
         'r', ...
         'FaceAlpha', 0.30, ...
         'EdgeColor', 'none');

    % Second integrated area
    t2 = t_switch(separationIdx:end);
    I2 = I_switch(separationIdx:end);

    fill([t2; flipud(t2)], ...
         [I2; zeros(size(I2))], ...
         'b', ...
         'FaceAlpha', 0.30, ...
         'EdgeColor', 'none');

    % Raw switching current
    plot(t_switch, I_switch, ...
         'k-', ...
         'LineWidth', 1.5);

    % Smoothed current
    plot(t_switch, I_smooth, ...
         '-', ...
         'Color', [0.5 0.5 0.5], ...
         'LineWidth', 1.0);

    % ALL derivative crossover points
    for k = 1:length(tCross)

        if k == crossIdx
            xline(tCross(k), ...
                'm-', ...
                'LineWidth', 3);
        else
            xline(tCross(k), ...
                'm--', ...
                'LineWidth', 1);
        end

    end

    % ALL local maxima
    plot(tPeaks, pks, ...
        'ko', ...
        'MarkerFaceColor', 'y', ...
        'MarkerSize', 6);

    % Selected local maximum
    plot(tPeaks(peakIdx), pks(peakIdx), ...
        'ro', ...
        'MarkerFaceColor', 'r', ...
        'MarkerSize', 8);

    % Zero-current line
    yline(0, 'k--');

    xlabel('Time');
    ylabel('Current (A)');

    title(sprintf('Switching Area Ratio = %.3f', ratio));

    grid on;
    box on;

    hold on
    yyaxis right 
    plot(t_switch, dIdt, 'b-', 'LineWidth', 1.5);
    hold on;
    yline(0, 'r--', 'LineWidth', 1.2);

    xlabel('Time');
    ylabel('dI/dt');


end