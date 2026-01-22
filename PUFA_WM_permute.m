clear all
close all

%% -------------------------------------------------
% Load data
%% -------------------------------------------------

db1 = readtable('/Users/tinadamkristensen/Desktop/COPSYCH_WM_PUFA/Data/merged_COPSYCH_WM_DTI_FBA_2.xlsx');

db1.Group  = categorical(db1.int_fish);
db1.NU_sex = categorical(db1.NU_sex);
db1.NU_age = double(db1.NU_age);

db1.NU_Tx_motion = double(db1.NU_Tx_motion);
db1.NU_Ty_motion = double(db1.NU_Ty_motion);
db1.NU_Tz_motion = double(db1.NU_Tz_motion);
db1.NU_Rx_motion = double(db1.NU_Rx_motion);
db1.NU_Ry_motion = double(db1.NU_Ry_motion);
db1.NU_Rz_motion = double(db1.NU_Rz_motion);

%% -------------------------------------------------
% Identify ROIs
%% -------------------------------------------------

roiMask = contains(db1.Properties.VariableNames, 'FA_');
roiVars = db1.Properties.VariableNames(roiMask);
nROI    = numel(roiVars);

%% -------------------------------------------------
% Ensure SubjectID
%% -------------------------------------------------

if ~ismember('SubjectID', db1.Properties.VariableNames)
    db1.SubjectID = (1:height(db1)).';
end
db1.SubjectID = categorical(db1.SubjectID);

%% -------------------------------------------------
% Reshape to long format
%% -------------------------------------------------

nSubj = height(db1);

Long = table;
Long.SubjectID = repmat(db1.SubjectID, nROI, 1);
Long.Group     = repmat(db1.Group,     nROI, 1);
Long.NU_age    = repmat(db1.NU_age,    nROI, 1);
Long.NU_sex    = repmat(db1.NU_sex,    nROI, 1);

Long.NU_Tx_motion = repmat(db1.NU_Tx_motion, nROI, 1);
Long.NU_Ty_motion = repmat(db1.NU_Ty_motion, nROI, 1);
Long.NU_Tz_motion = repmat(db1.NU_Tz_motion, nROI, 1);
Long.NU_Rx_motion = repmat(db1.NU_Rx_motion, nROI, 1);
Long.NU_Ry_motion = repmat(db1.NU_Ry_motion, nROI, 1);
Long.NU_Rz_motion = repmat(db1.NU_Rz_motion, nROI, 1);

ROIcol = strings(nSubj*nROI,1);
Ycol   = nan(nSubj*nROI,1);

idx = 1;
for r = 1:nROI
    vals = db1.(roiVars{r});
    Ycol(idx:idx+nSubj-1)   = vals;
    ROIcol(idx:idx+nSubj-1) = roiVars{r};
    idx = idx + nSubj;
end

Long.ROI = categorical(ROIcol);
Long.ROI = reordercats(Long.ROI, roiVars);
Long.FA  = Ycol;

%% -------------------------------------------------
% Fit mixed-effects model
%% -------------------------------------------------

fullModel = fitlme(Long, ...
    'FA ~ Group*ROI + NU_age + NU_sex + NU_Tx_motion + NU_Ty_motion + NU_Tz_motion + NU_Rx_motion + NU_Ry_motion + NU_Rz_motion + (1|SubjectID)');

%% -------------------------------------------------
% GLOBAL (OMNIBUS) TEST: Group effect across all ROIs
%% -------------------------------------------------

coefNames = fullModel.CoefficientNames;
nCoef     = numel(coefNames);

isGroupCoef   = startsWith(coefNames, 'Group');
idxGroupCoefs = find(isGroupCoef);

C = zeros(numel(idxGroupCoefs), nCoef);
for i = 1:numel(idxGroupCoefs)
    C(i, idxGroupCoefs(i)) = 1;
end

[pGlobal, FGlobal, DF1, DF2] = coefTest(fullModel, C);

fprintf('\nGLOBAL fish oil effect across all ROIs:\n');
fprintf('F(%d, %.1f) = %.3f, p = %.4g\n', DF1, DF2, FGlobal, pGlobal);

%% -------------------------------------------------
% Global effect size (partial R²)
%% -------------------------------------------------

chi2 = FGlobal * DF1;
partialR2 = chi2 / (chi2 + DF2);

fprintf('Global partial R^2 = %.4f\n', partialR2);

%% -------------------------------------------------
% PERMUTATION-BASED GLOBAL TEST
%% -------------------------------------------------

fprintf('\n--- PERMUTATION-BASED GLOBAL TEST ---\n');

nPerm = 1000;  % Number of permutations
fprintf('Running %d permutations...\n', nPerm);

% Store observed F-statistic
F_observed = FGlobal;

% Initialize permutation distribution
F_perm = nan(nPerm, 1);

% Get unique subjects for permutation
uniqueSubjects = unique(Long.SubjectID);
nUniqueSubj = numel(uniqueSubjects);

% Create a copy of Long for permutation
LongPerm = Long;

for perm = 1:nPerm
    if mod(perm, 100) == 0
        fprintf('  Permutation %d/%d\n', perm, nPerm);
    end
    
    % Permute group labels at the subject level
    permIdx = randperm(nUniqueSubj);
    groupPerm = db1.Group(permIdx);
    
    % Apply permuted labels to long format
    for s = 1:nUniqueSubj
        subjMask = LongPerm.SubjectID == uniqueSubjects(s);
        LongPerm.Group(subjMask) = groupPerm(s);
    end
    
    % Fit model with permuted data
    try
        permModel = fitlme(LongPerm, ...
            'FA ~ Group*ROI + NU_age + NU_sex + NU_Tx_motion + NU_Ty_motion + NU_Tz_motion + NU_Rx_motion + NU_Ry_motion + NU_Rz_motion + (1|SubjectID)', ...
            'Verbose', 0);
        
        % Test same hypothesis
        [~, F_perm(perm), ~, ~] = coefTest(permModel, C);
    catch
        F_perm(perm) = NaN;
    end
end

% Calculate permutation p-value
pGlobal_perm = mean(F_perm >= F_observed);

fprintf('\nPermutation-based global test results:\n');
fprintf('Observed F = %.3f\n', F_observed);
fprintf('Permutation p-value = %.4f\n', pGlobal_perm);
fprintf('(Based on %d permutations)\n', nPerm);

% Visualize permutation distribution
figure;
histogram(F_perm, 50, 'FaceColor', [0.7 0.7 0.7]);
hold on;
xline(F_observed, 'r--', 'LineWidth', 2, 'Label', 'Observed F');
xlabel('F-statistic');
ylabel('Frequency');
title('Permutation Distribution of Global F-statistic');
legend('Permutation null', 'Observed');

%% -------------------------------------------------
% ROI-specific group effects WITH CONFIDENCE INTERVALS
%% -------------------------------------------------

pVals       = nan(nROI,1);
Fstats      = nan(nROI,1);
betaVals    = nan(nROI,1);
betaCI_low  = nan(nROI,1);
betaCI_high = nan(nROI,1);
stdEffects  = nan(nROI,1);

idxGroupMain = find(strcmp(coefNames, 'Group_1'));

% Extract fixed effects and confidence intervals once
bFixed = fixedEffects(fullModel);
ciFixed = coefCI(fullModel);  % 95% confidence intervals

fprintf('\n--- ROI-SPECIFIC EFFECTS WITH CONFIDENCE INTERVALS ---\n');

for r = 1:nROI
    roiName = roiVars{r};
    c = zeros(1, nCoef);

    if r == 1
        c(idxGroupMain) = 1;
    else
        intPattern = ['Group_1:ROI_' roiName];
        idxInt = find(startsWith(coefNames, intPattern), 1);

        if isempty(idxInt)
            warning('Interaction term not found for %s', roiName);
            continue;
        end

        c(idxGroupMain) = 1;
        c(idxInt)       = 1;
    end

    [p,F,~,~] = coefTest(fullModel, c);
    beta     = c * bFixed;
    
    % Compute confidence interval for the contrast
    % CI for linear combination: c * β ± t * SE(c * β)
    % SE(c * β) = sqrt(c * Cov(β) * c')
    covB = fullModel.CoefficientCovariance;
    SE_beta = sqrt(c * covB * c');
    
    % Use t-distribution for CI
    alpha = 0.05;
    df = fullModel.DFE;
    t_crit = tinv(1 - alpha/2, df);
    
    CI_low  = beta - t_crit * SE_beta;
    CI_high = beta + t_crit * SE_beta;
    
    % Compute standardized effect size
    % For ROI-specific analysis, we'll compute Cohen's d-like measure
    % by dividing beta by the residual standard deviation
    residSD = fullModel.RMSE;
    stdEffect = beta / residSD;

    pVals(r)       = p;
    Fstats(r)      = F;
    betaVals(r)    = beta;
    betaCI_low(r)  = CI_low;
    betaCI_high(r) = CI_high;
    stdEffects(r)  = stdEffect;

    fprintf('%s   F = %.3f   p = %.4f   beta = %.5f [%.5f, %.5f]   std.effect = %.3f\n', ...
        roiName, F, p, beta, CI_low, CI_high, stdEffect);
end

%% -------------------------------------------------
% Summary of ROI-level effects
%% -------------------------------------------------

fprintf('\nROI-level summary:\n');
fprintf('Mean beta = %.5f\n', mean(betaVals, 'omitnan'));
fprintf('SD beta   = %.5f\n', std(betaVals,  'omitnan'));
fprintf('Mean standardized effect = %.3f\n', mean(stdEffects, 'omitnan'));
fprintf('SD standardized effect   = %.3f\n', std(stdEffects,  'omitnan'));

% Histogram of raw betas
figure;
histogram(betaVals, 20);
xlabel('ROI-specific Group beta');
ylabel('Count');
title('Distribution of fish oil effects across ROIs');

% Histogram of standardized effects
figure;
histogram(stdEffects, 20);
xlabel('Standardized Effect Size');
ylabel('Count');
title('Distribution of Standardized Effects across ROIs');

%% -------------------------------------------------
% Forest plot with confidence intervals
%% -------------------------------------------------

figure;
validIdx = ~isnan(betaVals);
validROIs = roiVars(validIdx);
validBetas = betaVals(validIdx);
validCI_low = betaCI_low(validIdx);
validCI_high = betaCI_high(validIdx);
nValidROI = sum(validIdx);

% Create error bars
errorbar(validBetas, 1:nValidROI, ...
    validBetas - validCI_low, validCI_high - validBetas, ...
    'o', 'horizontal', 'LineWidth', 1.5, 'MarkerSize', 6);
hold on;
xline(0, 'k--', 'LineWidth', 1);
yticks(1:nValidROI);
yticklabels(validROIs);
xlabel('Beta Coefficient (95% CI)');
ylabel('ROI');
title('Forest Plot: Fish Oil Effect on White Matter ROIs');
grid on;

%% -------------------------------------------------
% FDR correction (Benjamini–Hochberg)
%% -------------------------------------------------

valid    = ~isnan(pVals);
pValid   = pVals(valid);
roiValid = roiVars(valid);

% Extract valid values for beta, CI, and standardized effects
betaValid = betaVals(valid);
betaCI_low_valid = betaCI_low(valid);
betaCI_high_valid = betaCI_high(valid);
stdEffects_valid = stdEffects(valid);

m = numel(pValid);
[sortedP, sortIdx] = sort(pValid);
r = (1:m)';

bh = sortedP .* m ./ r;
bh = flipud(cummin(flipud(bh)));
bh(bh > 1) = 1;

pFDR = nan(size(pValid));
pFDR(sortIdx) = bh;

%% -------------------------------------------------
% Print significant ROIs
%% -------------------------------------------------

alpha = 0.05;
sigIdx = find(pFDR < alpha);

fprintf('\nFDR-significant ROIs (alpha = %.2f):\n', alpha);
if isempty(sigIdx)
    fprintf('  No ROIs survived FDR correction.\n');
else
    for k = 1:numel(sigIdx)
        i = sigIdx(k);
        fprintf('%s   raw p = %.4g   FDR p = %.4g   beta = %.5f [%.5f, %.5f]   std.effect = %.3f\n', ...
            roiValid{i}, pValid(i), pFDR(i), ...
            betaValid(i), betaCI_low_valid(i), betaCI_high_valid(i), ...
            stdEffects_valid(i));
    end
end

%% -------------------------------------------------
% Summary table export
%% -------------------------------------------------

resultsTable = table(roiVars', betaVals, betaCI_low, betaCI_high, stdEffects, Fstats, pVals, ...
    'VariableNames', {'ROI', 'Beta', 'CI_Lower', 'CI_Upper', 'Standardized_Effect', 'F_statistic', 'p_value'});

% Add FDR-corrected p-values
pFDR_full = nan(nROI, 1);
pFDR_full(valid) = pFDR;
resultsTable.pFDR = pFDR_full;

fprintf('\nResults table created with %d ROIs.\n', height(resultsTable));
fprintf('To save results, use: writetable(resultsTable, ''fishoil_results.csv'');\n');

fprintf('\n=== ANALYSIS COMPLETE ===\n');
