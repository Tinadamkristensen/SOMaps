%% Example/Demo Script: Key Features of Enhanced Fish Oil Analysis
% This script demonstrates the three major enhancements without requiring real data
% Use this to understand the concepts before running the full analysis

clear all
close all

fprintf('=== DEMONSTRATION OF ENHANCED FEATURES ===\n\n');

%% Feature 1: Confidence Intervals for Beta Coefficients
fprintf('--- FEATURE 1: CONFIDENCE INTERVALS ---\n');
fprintf('Computing 95%% confidence intervals for beta coefficients\n\n');

% Example: Simulate a simple beta coefficient and its standard error
beta_example = 0.0123;
SE_example = 0.0045;
df = 100;  % degrees of freedom
alpha = 0.05;

% Compute confidence interval
t_crit = tinv(1 - alpha/2, df);
CI_low = beta_example - t_crit * SE_example;
CI_high = beta_example + t_crit * SE_example;

fprintf('Example ROI:\n');
fprintf('  Beta coefficient: %.4f\n', beta_example);
fprintf('  Standard Error: %.4f\n', SE_example);
fprintf('  95%% CI: [%.4f, %.4f]\n', CI_low, CI_high);
fprintf('  Interpretation: ');
if CI_low > 0
    fprintf('Significantly POSITIVE effect (CI excludes 0)\n');
elseif CI_high < 0
    fprintf('Significantly NEGATIVE effect (CI excludes 0)\n');
else
    fprintf('NOT significant (CI includes 0)\n');
end

%% Feature 2: Standardized Effect Sizes
fprintf('\n--- FEATURE 2: STANDARDIZED EFFECT SIZES ---\n');
fprintf('Computing Cohen''s d-like standardized effects\n\n');

% Example: Convert raw beta to standardized effect
residSD_example = 0.0527;  % Residual standard deviation from model
std_effect = beta_example / residSD_example;

fprintf('Example calculation:\n');
fprintf('  Raw beta: %.4f\n', beta_example);
fprintf('  Residual SD: %.4f\n', residSD_example);
fprintf('  Standardized effect: %.3f\n', std_effect);
fprintf('  Interpretation: ');
if abs(std_effect) < 0.2
    fprintf('Negligible to small effect\n');
elseif abs(std_effect) < 0.5
    fprintf('Small to medium effect\n');
elseif abs(std_effect) < 0.8
    fprintf('Medium to large effect\n');
else
    fprintf('Large to very large effect\n');
end

%% Feature 3: Permutation-Based Global Test
fprintf('\n--- FEATURE 3: PERMUTATION TEST ---\n');
fprintf('Demonstrating permutation test concept\n\n');

% Simulate observed F-statistic
F_observed = 3.456;

% Simulate permutation distribution (normally would refit model each time)
nPerm = 100;  % Using fewer for demo speed
fprintf('Simulating %d permutations...\n', nPerm);

% In real analysis, these would come from refitting models
% Here we simulate from F-distribution for demonstration
rng(42);  % Set seed for reproducibility
F_perm = frnd(20, 100, nPerm, 1);  % Simulate null distribution

% Calculate permutation p-value
perm_p = mean(F_perm >= F_observed);

fprintf('  Observed F-statistic: %.3f\n', F_observed);
fprintf('  Permutation p-value: %.4f\n', perm_p);
fprintf('  Number of permutations: %d\n', nPerm);

% Visualize
figure('Position', [100 100 800 500]);
histogram(F_perm, 30, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'k');
hold on;
xline(F_observed, 'r--', 'LineWidth', 2.5);
text(F_observed + 0.3, max(histcounts(F_perm, 30)) * 0.9, ...
    sprintf('Observed F = %.2f', F_observed), ...
    'Color', 'r', 'FontWeight', 'bold', 'FontSize', 10);
xlabel('F-statistic', 'FontSize', 12);
ylabel('Frequency', 'FontSize', 12);
title('Permutation Distribution (Demonstration)', 'FontSize', 14);
legend({'Null Distribution', 'Observed F'}, 'Location', 'northeast');
grid on;

%% Demonstration: Forest Plot with Confidence Intervals
fprintf('\n--- DEMONSTRATION: FOREST PLOT ---\n');
fprintf('Creating example forest plot with confidence intervals\n\n');

% Simulate data for 10 example ROIs
nROI = 10;
roiNames = cell(nROI, 1);
for i = 1:nROI
    roiNames{i} = sprintf('ROI_%02d', i);
end

% Simulate beta coefficients and confidence intervals
rng(123);
betas = 0.01 + 0.008 * randn(nROI, 1);
SE = 0.004 + 0.001 * abs(randn(nROI, 1));
CI_lows = betas - 1.96 * SE;
CI_highs = betas + 1.96 * SE;

% Create forest plot
figure('Position', [100 100 900 600]);
errorbar(betas, 1:nROI, betas - CI_lows, CI_highs - betas, ...
    'o', 'horizontal', 'LineWidth', 1.5, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.2 0.4 0.8], 'Color', [0.2 0.4 0.8]);
hold on;
xline(0, 'k--', 'LineWidth', 1.5);

% Highlight significant effects
sig_idx = (CI_lows > 0) | (CI_highs < 0);
if any(sig_idx)
    plot(betas(sig_idx), find(sig_idx), 'r*', 'MarkerSize', 12, 'LineWidth', 2);
end

yticks(1:nROI);
yticklabels(roiNames);
xlabel('Beta Coefficient (95% CI)', 'FontSize', 12);
ylabel('Region of Interest', 'FontSize', 12);
title('Forest Plot: Effect Estimates with Confidence Intervals', 'FontSize', 14);
grid on;
xlim([min(CI_lows) - 0.005, max(CI_highs) + 0.005]);

fprintf('Forest plot created.\n');
fprintf('  Red stars (*) indicate statistically significant effects\n');
fprintf('  Error bars show 95%% confidence intervals\n');
fprintf('  Vertical dashed line at zero for reference\n');

%% Summary Table Example
fprintf('\n--- EXAMPLE RESULTS TABLE ---\n\n');

% Create example results table
resultsTable = table(roiNames, betas, CI_lows, CI_highs, betas ./ 0.05, ...
    'VariableNames', {'ROI', 'Beta', 'CI_Lower', 'CI_Upper', 'Std_Effect'});

% Add significance indicator
resultsTable.Significant = (CI_lows > 0) | (CI_highs < 0);

disp(resultsTable);

fprintf('\nTo export: writetable(resultsTable, ''results.csv'');\n');

%% Summary of Enhancements
fprintf('\n=== SUMMARY OF ENHANCEMENTS ===\n\n');
fprintf('1. CONFIDENCE INTERVALS\n');
fprintf('   - Provides uncertainty quantification for each beta\n');
fprintf('   - Based on coefficient covariance matrix\n');
fprintf('   - Uses t-distribution for proper inference\n\n');

fprintf('2. STANDARDIZED EFFECTS\n');
fprintf('   - Scale-free interpretation (Cohen''s d-like)\n');
fprintf('   - Comparable across ROIs and studies\n');
fprintf('   - Facilitates meta-analysis\n\n');

fprintf('3. PERMUTATION TEST\n');
fprintf('   - Non-parametric alternative to F-test\n');
fprintf('   - No distributional assumptions\n');
fprintf('   - Robust to violations of parametric assumptions\n\n');

fprintf('=== DEMONSTRATION COMPLETE ===\n');
fprintf('\nNext steps:\n');
fprintf('  1. Review the full analysis script: analyze_fishoil_white_matter.m\n');
fprintf('  2. Read the documentation: FISHOIL_ANALYSIS_README.md\n');
fprintf('  3. Check the quick reference: QUICK_REFERENCE.md\n');
fprintf('  4. Update the data path in the main script\n');
fprintf('  5. Run your analysis!\n');
