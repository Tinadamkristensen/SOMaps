# Quick Fix Guide

## The Problem
Your scatterplot shows different rho and p-values than your correlation matrix heatmap.

## The Solution
In your scatterplot section (around line 240), change this line:

### BEFORE (Incorrect):
```matlab
[r,p] = corr(X_res, Y_res);
```

### AFTER (Correct):
```matlab
[r, p] = partialcorr(X, Y, C);
```

## Explanation
- Your correlation matrix uses `partialcorr()` to control for age and sex
- Your scatterplot was using `corr()` on residuals, which gives different results
- Using `partialcorr(X, Y, C)` on the original data ensures both methods match

## Complete Fixed Scatterplot Section
Replace your entire scatterplot section with:

```matlab
%% Scatterplot: ROI vs cognitive function (partial regression)

% ==== USER-DEFINED VARIABLES ====
roiName = 'FA_Splenium_of_corpus_callosum';
cogName = 'CBCL_act';

% Extract variables
Y = Long.(roiName);          % ROI
X = Long.(cogName);          % Cognitive measure
C = [Long.NU_age, double(Long.NU_sex)];  % Covariates

% Remove rows with NaNs
validRows = ~any(isnan([Y X C]), 2);
Y = Y(validRows);
X = X(validRows);
C = C(validRows,:);

% Residualize ROI and cognitive variable w.r.t covariates
Y_res = Y - C * (C \ Y);
X_res = X - C * (C \ X);

% Fit linear model on residuals
mdl = fitlm(X_res, Y_res);

% Create prediction range
xFit = linspace(min(X_res), max(X_res), 100)';
[yFit, yCI] = predict(mdl, xFit);

% Plot
figure; hold on;

% Scatter points
scatter(X_res, Y_res, 40, 'filled', ...
    'MarkerFaceAlpha', 0.7);

% Confidence interval (lighter shade)
fill([xFit; flipud(xFit)], ...
     [yCI(:,1); flipud(yCI(:,2))], ...
     [0.7 0.7 0.7], ...
     'FaceAlpha', 0.4, ...
     'EdgeColor', 'none');

% Regression line
plot(xFit, yFit, 'k', 'LineWidth', 2);

% Labels and title
xlabel(strrep(cogName,'_','\_'));
ylabel(strrep(roiName,'_','\_'));
title(sprintf('Partial relationship: %s vs %s', ...
    strrep(roiName,'_','\_'), strrep(cogName,'_','\_')));

set(gca,'FontSize',12);
box on;

% FIX: Use partialcorr to match the correlation matrix calculation
% This computes the partial correlation controlling for age and sex
[r, p] = partialcorr(X, Y, C);

text(0.05, 0.95, ...
    sprintf('\\rho = %.2f, p = %.3f', r, p), ...
    'Units','normalized', ...
    'VerticalAlignment','top', ...
    'FontSize',11);

g = db1.Group(validRows);
gscatter(X_res, Y_res, g);
```

## Why This Works
The `partialcorr(X, Y, C)` function computes the exact same partial correlation as used in your correlation matrix, ensuring consistency between the heatmap and scatterplot displays.
