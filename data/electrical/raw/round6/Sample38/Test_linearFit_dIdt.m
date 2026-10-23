%% Switching area ratio using linear interpolation of derivative extrema

% Smooth raw switching current
I_smooth = sgolayfilt(I_switch, 3, 21);

% Calculate derivative
dIdt = gradient(I_smooth, t_switch);

% ---------------------------------------------------------
% Search windows for derivative extrema
% ---------------------------------------------------------

maxWindow = [9.3e-6 9.5e-6];
minWindow = [9.7e-6 9.9e-6];

idxMin = t_switch >= minWindow(1) & t_switch <= minWindow(2);
idxMax = t_switch >= maxWindow(1) & t_switch <= maxWindow(2);

if ~any(idxMin) || ~any(idxMax)

    warning('One or both derivative search windows contain no data.');

else

    % -----------------------------------------------------
    % Select single derivative minimum and maximum
    % -----------------------------------------------------

    [dMin, localMinIdx] = min(dIdt(idxMin));
    minIndices = find(idxMin);
    minIdx = minIndices(localMinIdx);

    tMin = t_switch(minIdx);

    [dMax, localMaxIdx] = max(dIdt(idxMax));
    maxIndices = find(idxMax);
    maxIdx = maxIndices(localMaxIdx);

    tMax = t_switch(maxIdx);

    % -----------------------------------------------------
    % Find ALL derivative zero crossings
    % -----------------------------------------------------

    zeroIdx = find(dIdt(1:end-1) > 0 & dIdt(2:end) <= 0);

    tCross = zeros(size(zeroIdx));

    for k = 1:length(zeroIdx)

        i = zeroIdx(k);

        tCross(k) = t_switch(i) - ...
            dIdt(i) * (t_switch(i+1)-t_switch(i)) / ...
            (dIdt(i+1)-dIdt(i));

    end

    % Only show crossovers in the region of interest
    validCross = tCross >= 9.5e-6 & tCross <= 9.7e-6;
    tCrossWindow = tCross(validCross);

    % -----------------------------------------------------
    % Linear fit between derivative extrema
    % -----------------------------------------------------

    m = (dMax - dMin) / (tMax - tMin);
    b = dMin - m*tMin;

    % dI/dt = 0
    separationTime = -b/m;

    % Corresponding raw-data index
    separationIdx = find(t_switch >= separationTime, 1);

    % -----------------------------------------------------
    % Integrate RAW switching current
    % -----------------------------------------------------

    A1 = abs(trapz(t_switch(1:separationIdx), ...
                   I_switch(1:separationIdx)));

    A2 = abs(trapz(t_switch(separationIdx:end), ...
                   I_switch(separationIdx:end)));

    % Area ratio
    ratio = A1/A2;

    % Fractions
    fraction1 = A1/(A1+A2);
    fraction2 = A2/(A1+A2);

    % -----------------------------------------------------
    % Print results
    % -----------------------------------------------------

    fprintf('\nDerivative minimum:\n');
    fprintf('Time = %.4f us\n', tMin*1e6);
    fprintf('dI/dt = %.4e A/s\n', dMin);

    fprintf('\nDerivative maximum:\n');
    fprintf('Time = %.4f us\n', tMax*1e6);
    fprintf('dI/dt = %.4e A/s\n', dMax);

    fprintf('\nDetected derivative crossovers (9.5–9.7 us):\n');

    if isempty(tCrossWindow)
        fprintf('None detected.\n');
    else
        fprintf('%.4f us\n', tCrossWindow*1e6);
    end

    fprintf('\nLinear-fit zero crossing:\n');
    fprintf('Separation = %.4f us\n', separationTime*1e6);

    fprintf('\nArea 1 = %.4e A*s\n', A1);
    fprintf('Area 2 = %.4e A*s\n', A2);

    fprintf('\nArea ratio A1/A2 = %.4f\n', ratio);

    fprintf('Fraction 1 = %.2f %%\n', fraction1*100);
    fprintf('Fraction 2 = %.2f %%\n', fraction2*100);

    % -----------------------------------------------------
    % Plot
    % -----------------------------------------------------

    figure;
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
    % SELECTED SEPARATION POINT
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
    xline(minWindow(1), 'r:', 'LineWidth', 1);
    xline(minWindow(2), 'r:', 'LineWidth', 1);

    xline(maxWindow(1), 'g:', 'LineWidth', 1);
    xline(maxWindow(2), 'g:', 'LineWidth', 1);

    % Zero-current line
    yyaxis left
    yline(0, 'k--');

    % Labels
    xlabel('Time');

    title(sprintf('Switching Area Ratio = %.3f', ratio));

    grid on;
    box on;

end