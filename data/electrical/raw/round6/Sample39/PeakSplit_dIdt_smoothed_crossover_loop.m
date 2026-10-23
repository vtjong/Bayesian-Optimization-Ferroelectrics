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
%   5. Applies stronger smoothing before calculating dI/dt
%   6. Finds positive-to-negative dI/dt zero crossings in the target window
%   7. Selects the crossover with the highest smoothed-current value
%   8. Integrates the RAW switching current on either side
%   9. Calculates the area ratio and normalized fractions
%  10. Plots raw/smoothed current, derivative, candidate crossovers, and areas
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

% Search window for the separation point
crossWindow = [9.5e-6 9.7e-6];

% Stronger smoothing than the previous 21-point window
sgolayOrder = 3;
sgolayFrame = 41;

%% Preallocate results

results = table( ...
    strings(length(files),1), ...
    NaN(length(files),1), ...
    NaN(length(files),1), ...
    NaN(length(files),1), ...
    NaN(length(files),1), ...
    NaN(length(files),1), ...
    NaN(length(files),1), ...
    'VariableNames', { ...
    'FileName', ...
    'Separation_us', ...
    'Area1_As', ...
    'Area2_As', ...
    'AreaRatio', ...
    'Fraction1', ...
    'Fraction2'});

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

        %% Stronger smoothing of raw switching current

        I_smooth = sgolayfilt(I_switch, sgolayOrder, sgolayFrame);

        %% Calculate derivative from smoothed current

        dIdt = gradient(I_smooth, t_switch);

        %% Find positive -> negative derivative zero crossings
        % These correspond to local maxima in the smoothed current.

        zeroIdx = find(dIdt(1:end-1) > 0 & dIdt(2:end) <= 0);

        tCross = zeros(size(zeroIdx));

        for k = 1:length(zeroIdx)
            i = zeroIdx(k);
            tCross(k) = t_switch(i) - ...
                dIdt(i) * (t_switch(i+1)-t_switch(i)) / ...
                (dIdt(i+1)-dIdt(i));
        end

        % Keep only crossovers in the target window
        validCross = tCross >= crossWindow(1) & ...
                     tCross <= crossWindow(2);
        tCrossWindow = tCross(validCross);

        %% Select crossover with highest smoothed-current value

        if isempty(tCrossWindow)
            error('No positive-to-negative derivative crossover found between 9.5 and 9.7 us.');
        end

        IatCross = interp1(t_switch, I_smooth, tCrossWindow, 'linear');
        [~, selectedCrossIdx] = max(IatCross);
        separationTime = tCrossWindow(selectedCrossIdx);

        separationIdx = find(t_switch >= separationTime, 1);
        if isempty(separationIdx)
            error('Could not determine separation index.');
        end

        %% Integrate RAW switching current

        A1 = abs(trapz(t_switch(1:separationIdx), ...
                       I_switch(1:separationIdx)));
        A2 = abs(trapz(t_switch(separationIdx:end), ...
                       I_switch(separationIdx:end)));

        %% Area ratio and fractions

        ratio = A1 / A2;
        fraction1 = A1 / (A1 + A2);
        fraction2 = A2 / (A1 + A2);

        %% Store results

        results.FileName(fileNum) = string(filename);
        results.Separation_us(fileNum) = separationTime * 1e6;
        results.Area1_As(fileNum) = A1;
        results.Area2_As(fileNum) = A2;
        results.AreaRatio(fileNum) = ratio;
        results.Fraction1(fileNum) = fraction1;
        results.Fraction2(fileNum) = fraction2;

        %% Print results

        fprintf('\nDetected derivative crossovers (9.5-9.7 us):\n');
        fprintf('  %.4f us\n', tCrossWindow*1e6);
        fprintf('\nSelected derivative crossover:\n');
        fprintf('  Separation = %.4f us\n', separationTime*1e6);
        fprintf('  Smoothed current at crossover = %.4e A\n', ...
            interp1(t_switch, I_smooth, separationTime, 'linear'));
        fprintf('\nArea 1 = %.4e A*s\n', A1);
        fprintf('Area 2 = %.4e A*s\n', A2);
        fprintf('Area ratio A1/A2 = %.4f\n', ratio);
        fprintf('Fraction 1 = %.2f %%\n', fraction1*100);
        fprintf('Fraction 2 = %.2f %%\n', fraction2*100);

        %% Plot

        figure('Name', filename, 'NumberTitle', 'off');
        hold on;

        % LEFT AXIS: CURRENT
        yyaxis left

        t1 = t_switch(1:separationIdx);
        I1 = I_switch(1:separationIdx);
        fill([t1; flipud(t1)], [I1; zeros(size(I1))], 'r', ...
             'FaceAlpha', 0.30, 'EdgeColor', 'none');

        t2 = t_switch(separationIdx:end);
        I2 = I_switch(separationIdx:end);
        fill([t2; flipud(t2)], [I2; zeros(size(I2))], 'b', ...
             'FaceAlpha', 0.30, 'EdgeColor', 'none');

        plot(t_switch, I_switch, 'k-', 'LineWidth', 1.5);
        plot(t_switch, I_smooth, '-', ...
             'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
        ylabel('Current (A)');

        % RIGHT AXIS: DERIVATIVE
        yyaxis right
        plot(t_switch, dIdt, 'b-', 'LineWidth', 1.2);
        hold on;
        yline(0, 'k--', 'LineWidth', 1);
        ylabel('dI/dt (A/s)');

        % All candidate crossovers; selected one is thicker
        for k = 1:length(tCrossWindow)
            if k == selectedCrossIdx
                xline(tCrossWindow(k), 'm-', 'LineWidth', 3);
            else
                xline(tCrossWindow(k), 'm:', 'LineWidth', 1.2);
            end
        end

        plot(separationTime, 0, 'mo', ...
             'MarkerFaceColor', 'm', 'MarkerSize', 7);

        % Search-window boundaries
        xline(crossWindow(1), 'r:', 'LineWidth', 1.5);
        xline(crossWindow(2), 'r:', 'LineWidth', 1.5);

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

