%% PUND peak-splitting analysis for all PUND .xls files
%
% Finds all .xls files in the current MATLAB folder containing "PUND"
% in the filename, then runs the peak-splitting analysis on each file.
%
% The analysis:
%   1. Reads the PUND data from Sheet "Data", B2:D3401
%   2. Extracts the switching pulse (indices 1800:2000)
%   3. Smooths the raw switching current
%   4. Calculates dI/dt
%   5. Finds the single minimum and maximum dI/dt in the specified windows
%   6. Fits a line between those extrema
%   7. Finds the dI/dt = 0 crossover of that fitted line
%   8. Integrates the RAW switching current on either side
%   9. Calculates the area ratio and normalized fractions
%  10. Plots the current, derivative, extrema, fit, crossovers, and areas
%
% Results for all files are stored in the "results" table at the end.

clear;
clc;
close all;

%% Find PUND files

files = dir('*PUND*.xls');

if isempty(files)
    error('No .xls files containing "PUND" were found in the current folder.');
end

fprintf('Found %d PUND file(s).\n\n', length(files));

%% Analysis settings

I_switch_idx = 1800:2000;

% Derivative extrema search windows
maxWindow = [9.3e-6 9.5e-6];
minWindow = [9.7e-6 9.9e-6];

% Region in which to display derivative zero crossings
crossWindow = [9.5e-6 9.7e-6];

% Smoothing parameters
sgolayOrder = 3;
sgolayFrame = 21;

%% Preallocate results

results = table( ...
    strings(length(files),1), ...
    NaN(length(files),1), ...
    NaN(length(files),1), ...
    NaN(length(files),1), ...
    NaN(length(files),1), ...
    NaN(length(files),1), ...
    NaN(length(files),1), ...
    NaN(length(files),1), ...
    NaN(length(files),1), ...
    NaN(length(files),1), ...
    'VariableNames', { ...
    'FileName', ...
    't_dIdt_min_us', ...
    'dIdt_min_A_per_s', ...
    't_dIdt_max_us', ...
    'dIdt_max_A_per_s', ...
    'Separation_us', ...
    'Area1_As', ...
    'Area2_As', ...
    'AreaRatio', ...
    'Fraction1'});

results.Fraction2 = NaN(height(results),1);

%% Loop through all PUND files

for fileNum = 1:length(files)

    filename = files(fileNum).name;

    fprintf('\n============================================\n');
    fprintf('Processing %d/%d: %s\n', fileNum, length(files), filename);
    fprintf('============================================\n');

    try

        %% Load data

        pund = readmatrix(filename, ...
            'Sheet', 'Data', ...
            'Range', 'B2:D3401');

        t = pund(:,3);
        I = pund(:,2);
        V = pund(:,1);

        %% Extract switching current

        t_switch = t(I_switch_idx);
        I_switch = I(I_switch_idx);

        %% Check data

        if length(I_switch) ~= length(I_switch_idx)
            error('Switching-current range does not contain the expected number of points.');
        end

        %% Smooth raw switching current

        I_smooth = sgolayfilt(I_switch, sgolayOrder, sgolayFrame);

        %% Calculate derivative

        dIdt = gradient(I_smooth, t_switch);

        %% Find derivative minimum and maximum

        idxMin = t_switch >= minWindow(1) & t_switch <= minWindow(2);
        idxMax = t_switch >= maxWindow(1) & t_switch <= maxWindow(2);

        if ~any(idxMin) || ~any(idxMax)
            error('One or both derivative search windows contain no data.');
        end

        % Most negative dI/dt in the first window
        [dMin, localMinIdx] = min(dIdt(idxMin));
        minIndices = find(idxMin);
        minIdx = minIndices(localMinIdx);
        tMin = t_switch(minIdx);

        % Most positive dI/dt in the second window
        [dMax, localMaxIdx] = max(dIdt(idxMax));
        maxIndices = find(idxMax);
        maxIdx = maxIndices(localMaxIdx);
        tMax = t_switch(maxIdx);

        %% Find all positive -> negative derivative zero crossings

        zeroIdx = find(dIdt(1:end-1) > 0 & dIdt(2:end) <= 0);

        tCross = zeros(size(zeroIdx));

        for k = 1:length(zeroIdx)

            i = zeroIdx(k);

            tCross(k) = t_switch(i) - ...
                dIdt(i) * (t_switch(i+1)-t_switch(i)) / ...
                (dIdt(i+1)-dIdt(i));

        end

        % Keep only crossovers in the region of interest
        validCross = tCross >= crossWindow(1) & ...
                     tCross <= crossWindow(2);

        tCrossWindow = tCross(validCross);

        %% Linear fit between derivative extrema

        m = (dMax - dMin) / (tMax - tMin);
        b = dMin - m*tMin;

        % dI/dt = 0
        separationTime = -b/m;

        %% Check separation point

        if separationTime < min(tMin,tMax) || ...
           separationTime > max(tMin,tMax)

            warning('%s: Linear-fit crossover lies outside the two extrema.', filename);

        end

        separationIdx = find(t_switch >= separationTime, 1);

        if isempty(separationIdx)
            error('Could not determine separation index.');
        end

        %% Integrate RAW switching current

        A1 = abs(trapz( ...
            t_switch(1:separationIdx), ...
            I_switch(1:separationIdx)));

        A2 = abs(trapz( ...
            t_switch(separationIdx:end), ...
            I_switch(separationIdx:end)));

        %% Area ratio and fractions

        ratio = A1 / A2;

        fraction1 = A1 / (A1 + A2);
        fraction2 = A2 / (A1 + A2);

        %% Store results

        results.FileName(fileNum) = string(filename);
        results.t_dIdt_min_us(fileNum) = tMin * 1e6;
        results.dIdt_min_A_per_s(fileNum) = dMin;
        results.t_dIdt_max_us(fileNum) = tMax * 1e6;
        results.dIdt_max_A_per_s(fileNum) = dMax;
        results.Separation_us(fileNum) = separationTime * 1e6;
        results.Area1_As(fileNum) = A1;
        results.Area2_As(fileNum) = A2;
        results.AreaRatio(fileNum) = ratio;
        results.Fraction1(fileNum) = fraction1;
        results.Fraction2(fileNum) = fraction2;

        %% Print results

        fprintf('\nDerivative minimum:\n');
        fprintf('  Time = %.4f us\n', tMin*1e6);
        fprintf('  dI/dt = %.4e A/s\n', dMin);

        fprintf('\nDerivative maximum:\n');
        fprintf('  Time = %.4f us\n', tMax*1e6);
        fprintf('  dI/dt = %.4e A/s\n', dMax);

        fprintf('\nDetected derivative crossovers (9.5-9.7 us):\n');

        if isempty(tCrossWindow)
            fprintf('  None detected.\n');
        else
            fprintf('  %.4f us\n', tCrossWindow*1e6);
        end

        fprintf('\nLinear-fit zero crossing:\n');
        fprintf('  Separation = %.4f us\n', separationTime*1e6);

        fprintf('\nArea 1 = %.4e A*s\n', A1);
        fprintf('Area 2 = %.4e A*s\n', A2);
        fprintf('Area ratio A1/A2 = %.4f\n', ratio);
        fprintf('Fraction 1 = %.2f %%\n', fraction1*100);
        fprintf('Fraction 2 = %.2f %%\n', fraction2*100);

        %% Plot

        figure('Name', filename, 'NumberTitle', 'off');
        hold on;

        % =====================================================
        % LEFT AXIS: CURRENT
        % =====================================================

        yyaxis left

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
             'Color', [0.5 0.5 0.5], ...
             'LineWidth', 1.0);

        ylabel('Current (A)');

        % =====================================================
        % RIGHT AXIS: DERIVATIVE
        % =====================================================

        yyaxis right

        % dI/dt
        plot(t_switch, dIdt, ...
             'b-', ...
             'LineWidth', 1.2);

        hold on;

        % Derivative minimum
        plot(tMin, dMin, ...
             'ro', ...
             'MarkerFaceColor', 'r', ...
             'MarkerSize', 8);

        % Derivative maximum
        plot(tMax, dMax, ...
             'go', ...
             'MarkerFaceColor', 'g', ...
             'MarkerSize', 8);

        % Linear fit between extrema
        tFit = linspace(tMin, tMax, 200);
        dFit = m*tFit + b;

        plot(tFit, dFit, ...
             'm--', ...
             'LineWidth', 2);

        % Zero derivative line
        yline(0, ...
             'k--', ...
             'LineWidth', 1);

        ylabel('dI/dt (A/s)');

        % =====================================================
        % SEPARATION POINT
        % =====================================================

        % Thick line for linear-fit zero crossing
        xline(separationTime, ...
             'm-', ...
             'LineWidth', 3);

        % Plot all detected derivative crossovers
        for k = 1:length(tCrossWindow)

            xline(tCrossWindow(k), ...
                  'm:', ...
                  'LineWidth', 1.2);

        end

        % Search-window boundaries
        xline(minWindow(1), 'r:', 'LineWidth', 2);
        xline(minWindow(2), 'r:', 'LineWidth', 2);

        xline(maxWindow(1), 'g:', 'LineWidth', 2);
        xline(maxWindow(2), 'g:', 'LineWidth', 2);

        % Zero-current line
        yyaxis left
        yline(0, 'k--');

        xlabel('Time');

        title(sprintf('%s | Area Ratio = %.3f', filename, ratio), ...
              'Interpreter', 'none');

        grid on;
        box on;

    catch ME

        fprintf('\n*** ERROR processing %s ***\n', filename);
        fprintf('%s\n', ME.message);

        results.FileName(fileNum) = string(filename);

    end

end

%% Display summary

fprintf('\n\n============================================\n');
fprintf('SUMMARY\n');
fprintf('============================================\n');

disp(results);

