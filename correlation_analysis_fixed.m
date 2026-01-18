% MATLAB Correlation Analysis Script with Partial Correlations
% This script computes partial correlations between ROI and cognitive variables,
% controlling for age and sex as covariates.
%
% IMPORTANT: Update the file path below to point to your data file
clear all
close all

db1 = readtable('/Users/tinadamkristensen/Desktop/COPSYCH_WM_PUFA/Data/merged_COPSYCH_WM_DTI_FBA_2_BEHAV.xlsx');

%% Preparation
db1.Group  = categorical(db1.int_fish);
db1.NU_sex = categorical(db1.NU_sex);
db1.NU_age = double(db1.NU_age);

%% Cognitive variables (ensure numeric)
db1.Functional_Level = double(db1.Functional_Level);
db1.General_Index = double(db1.General_Index);
db1.Verbal_Working_Memory_Index = double(db1.Verbal_Working_Memory_Index);
db1.CBCL_Total_problem = double(db1.CBCL_Total_problem);
db1.CBCL_ext = double(db1.CBCL_ext);
db1.CBCL_int = double(db1.CBCL_int);
db1.CBCL_dep = double(db1.CBCL_dep);
db1.CBCL_anx = double(db1.CBCL_anx);
db1.CBCL_som = double(db1.CBCL_som);
db1.CBCL_adhd = double(db1.CBCL_adhd);
db1.CBCL_opd = double(db1.CBCL_opd);
db1.CBCL_conduct = double(db1.CBCL_conduct);
db1.CBCL_act = double(db1.CBCL_act);
db1.CBCL_soc = double(db1.CBCL_soc);
db1.CBCL_sch = double(db1.CBCL_sch);
db1.CBCL_TOTAL = double(db1.CBCL_TOTAL);
db1.ADHDRS_opmaerksomhed_sc_F = double(db1.ADHDRS_opmaerksomhed_sc_F);
db1.ADHDRS_impuls_hyperakt_sc_F = double(db1.ADHDRS_impuls_hyperakt_sc_F);
db1.ADHDRS_adfaerdsforstyrrelse_sc_F = double(db1.ADHDRS_adfaerdsforstyrrelse_sc_F);
db1.ADHDRS_ADHD_CD_symptoms_sc_F = double(db1.ADHDRS_ADHD_CD_symptoms_sc_F);
db1.ADHDRS_ADHD_symptoms_sc_F = double(db1.ADHDRS_ADHD_symptoms_sc_F);
db1.Planning_Thinking_Time = double(db1.Planning_Thinking_Time);
db1.Spatial_Working_Memory = double(db1.Spatial_Working_Memory);
db1.Cognitive_Flexibility = double(db1.Cognitive_Flexibility);
db1.Inhibition = double(db1.Inhibition);
db1.Strategy = double(db1.Strategy);
db1.Planning = double(db1.Planning);
db1.Processing_Speed_Index = double(db1.Processing_Speed_Index);
db1.Attention = double(db1.Attention);
db1.Motor_Function = double(db1.Motor_Function);
db1.Verbal_Memory = double(db1.Verbal_Memory);
db1.Visual_Memory = double(db1.Visual_Memory);
db1.BRIEF_Inhibition = double(db1.BRIEF_Inhibition);
db1.BRIEF_Planning = double(db1.BRIEF_Planning);
db1.BRIEF_GEF = double(db1.BRIEF_GEF);
db1.BRIEF_Behave_Index = double(db1.BRIEF_Behave_Index);
db1.BRIEF_Emo_Index = double(db1.BRIEF_Emo_Index);
db1.BRIEF_Cogn_Index = double(db1.BRIEF_Cogn_Index);
db1.RTI_Movement_Time = double(db1.RTI_Movement_Time);
db1.RTI_Reaction_Time = double(db1.RTI_Reaction_Time);
db1.SWM_Between_Errors = double(db1.SWM_Between_Errors);

%% Cognitive variable list
cognitiveVars = {
    'Functional_Level'
    'General_Index'
    'Verbal_Working_Memory_Index'
    'CBCL_Total_problem'
    'CBCL_ext'
    'CBCL_int'
    'CBCL_dep'
    'CBCL_anx'
    'CBCL_som'
    'CBCL_adhd'
    'CBCL_opd'
    'CBCL_conduct'
    'CBCL_act'
    'CBCL_soc'
    'CBCL_sch'
    'CBCL_TOTAL'
    'ADHDRS_opmaerksomhed_sc_F'
    'ADHDRS_impuls_hyperakt_sc_F'
    'ADHDRS_adfaerdsforstyrrelse_sc_F'
    'ADHDRS_ADHD_CD_symptoms_sc_F'
    'ADHDRS_ADHD_symptoms_sc_F'
    'Planning_Thinking_Time'
    'Spatial_Working_Memory'
    'Cognitive_Flexibility'
    'Inhibition'
    'Strategy'
    'Planning'
    'Processing_Speed_Index'
    'Attention'
    'Motor_Function'
    'Verbal_Memory'
    'Visual_Memory'
    'BRIEF_Inhibition'
    'BRIEF_Planning'
    'BRIEF_GEF'
    'BRIEF_Behave_Index'
    'BRIEF_Emo_Index'
    'BRIEF_Cogn_Index'
    'RTI_Movement_Time'
    'RTI_Reaction_Time'
    'SWM_Between_Errors'
};

%% Mean imputation
for v = 1:numel(cognitiveVars)
    x = db1.(cognitiveVars{v});
    db1.(cognitiveVars{v})(isnan(x)) = mean(x,'omitnan');
end

%% ROI data
roiVars = db1.Properties.VariableNames(contains(db1.Properties.VariableNames,'FA_'));
nROI = numel(roiVars);

dataArray = zeros(height(db1), nROI + numel(cognitiveVars));
dataArray(:,1:nROI) = table2array(db1(:,roiVars));
dataArray(:,nROI+1:end) = table2array(db1(:,cognitiveVars));

Long = array2table( ...
    dataArray, ...
    'VariableNames', [roiVars(:).' cognitiveVars(:).'] );

% Add covariates separately
Long.NU_age = db1.NU_age;
Long.NU_sex = db1.NU_sex;

%% Partial correlations
results = struct();

for i = 1:nROI
    roiData = Long{:,roiVars{i}};
    cognitiveData = Long{:,cognitiveVars};
    covariates = [Long.NU_age, double(Long.NU_sex)];

    validRows = ~any(isnan([roiData cognitiveData covariates]),2);

    [rho, pval] = partialcorr( ...
        roiData(validRows), ...
        cognitiveData(validRows,:), ...
        covariates(validRows,:) );

    results(i).ROI  = roiVars{i};
    results(i).rho  = rho;
    results(i).pval = pval;
end

%% Bubble plot: Correlation matrix with effect size and significance

nCog = numel(cognitiveVars);

correlationMatrix = NaN(nROI, nCog);
pMatrix           = NaN(nROI, nCog);

for i = 1:nROI
    correlationMatrix(i,:) = results(i).rho;
    pMatrix(i,:)           = results(i).pval;
end

alpha = 0.05;

roiLabels = strrep(roiVars, '_', '\_');
cogLabels = strrep(cognitiveVars, '_', '\_');

% ===========================================================
% FDR correction across all ROI × cognitive correlations
% ===========================================================

pvals_vec = pMatrix(:);                             % vectorize all p-values
qvals_vec = mafdr(pvals_vec, 'BHFDR', true);        % Benjamini–Hochberg
qMatrix   = reshape(qvals_vec, size(pMatrix));      % reshape back

% ===========================================================
% Create bubble plot with circles
% ===========================================================

figure('Position', [100, 100, 1200, 800]);
hold on;

% Get colormap for directionality (turbo colormap)
cmap = turbo(256);
cmap_center = 128;  % Center of colormap (near zero correlation)

% Maximum circle size (in points)
max_bubble_size = 1000;

% Plot each correlation as a circle
for i = 1:nROI
    for j = 1:nCog
        rho = correlationMatrix(i,j);
        p_val = pMatrix(i,j);
        q_val = qMatrix(i,j);
        
        % Circle size based on effect size (absolute correlation)
        bubble_size = abs(rho) * max_bubble_size;
        
        % Determine color intensity based on significance level
        if q_val < 0.05
            % FDR significant: full intensity (alpha = 1.0)
            alpha_val = 1.0;
        elseif p_val < 0.05
            % Raw significant: intermediate intensity (alpha = 0.5)
            alpha_val = 0.5;
        else
            % Non-significant: very transparent (alpha = 0.15)
            alpha_val = 0.15;
        end
        
        % Get color based on correlation direction
        % Map rho from [-0.5, 0.5] to colormap indices
        color_idx = round((rho + 0.5) / 1.0 * 255) + 1;
        color_idx = max(1, min(256, color_idx));  % clamp to valid range
        circle_color = cmap(color_idx, :);
        
        % Plot circle at position (j, i) with size and color
        scatter(j, i, bubble_size, circle_color, 'filled', ...
            'MarkerFaceAlpha', alpha_val, ...
            'MarkerEdgeColor', 'none');
    end
end

% Set axis properties
xlim([0.5, nCog + 0.5]);
ylim([0.5, nROI + 0.5]);
axis ij;  % Flip y-axis to match matrix convention
axis equal;
axis tight;

set(gca, ...
    'XTick', 1:nCog, ...
    'XTickLabel', cogLabels, ...
    'YTick', 1:nROI, ...
    'YTickLabel', roiLabels, ...
    'TickLabelInterpreter', 'tex', ...
    'FontSize', 8, ...
    'Box', 'on', ...
    'Layer', 'top');

xtickangle(45);

xlabel('Cognitive Measures');
ylabel('Regions of Interest');
title('Partial correlations: Circle size = effect size, Color = direction, Intensity = significance');

% Add colorbar for directionality
colormap(cmap);
caxis([-0.5 0.5]);
cb = colorbar;
cb.Label.String = 'Correlation (rho)';

% Add legend for significance levels
legend_x = nCog + 1;
legend_y_base = nROI * 0.2;
legend_spacing = nROI * 0.15;

% Example circles for legend
legend_size = 300;

% FDR significant
scatter(legend_x, legend_y_base, legend_size, [0.5 0.5 0.5], 'filled', ...
    'MarkerFaceAlpha', 1.0, 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
text(legend_x + 0.3, legend_y_base, 'q < 0.05 (FDR)', ...
    'FontSize', 8, 'VerticalAlignment', 'middle');

% Raw significant
scatter(legend_x, legend_y_base + legend_spacing, legend_size, [0.5 0.5 0.5], 'filled', ...
    'MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
text(legend_x + 0.3, legend_y_base + legend_spacing, 'p < 0.05 (raw)', ...
    'FontSize', 8, 'VerticalAlignment', 'middle');

% Non-significant
scatter(legend_x, legend_y_base + 2*legend_spacing, legend_size, [0.5 0.5 0.5], 'filled', ...
    'MarkerFaceAlpha', 0.15, 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
text(legend_x + 0.3, legend_y_base + 2*legend_spacing, 'n.s.', ...
    'FontSize', 8, 'VerticalAlignment', 'middle');

hold off;

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

% Get group data for coloring
g = db1.Group(validRows);

% Scatter points colored by group
gscatter(X_res, Y_res, g, [], [], 40, 'off');

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
legend('show');

% FIX: Use partialcorr to match the correlation matrix calculation
% This computes the partial correlation controlling for age and sex
[r, p] = partialcorr(X, Y, C);

text(0.05, 0.95, ...
    sprintf('\\rho = %.2f, p = %.3f', r, p), ...
    'Units','normalized', ...
    'VerticalAlignment','top', ...
    'FontSize',11);
