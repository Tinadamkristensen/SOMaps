% MATLAB Correlation Analysis Script with Bubble Plot Visualization
% This script computes partial correlations between ROI and cognitive variables,
% controlling for age and sex as covariates.
%
% IMPORTANT: Update the file path below to point to your data file
clear all
close all

db1 = readtable('/Users/tinadamkristensen/Desktop/SPECTRUM_PLS_2026/data_CORR_COGN/FINAL_SPECTRUM_CORR_sign_FEP_CON.xlsx', ...
                'VariableNamingRule','preserve');

%% Preparation
db1.Group  = categorical(db1.Group);
db1.NU_Sex = categorical(db1.NU_Sex);
db1.NU_Age = double(db1.NU_Age);

db1.COG_DART = double(db1.COG_DART);
%db1.COG_VIQ = double(db1.COG_VIQ);
%db1.COG_PIQ = double(db1.COG_PIQ);
db1.COG_Verbal_Memory = double(db1.COG_Verbal_Memory);
db1.COG_Verbal_Wmemory = double(db1.COG_Verbal_Wmemory);
db1.COG_Verbal_Fluency = double(db1.COG_Verbal_Fluency);
db1.COG_Processing_Speed = double(db1.COG_Processing_Speed);
db1.COG_ToL_Planning = double(db1.COG_ToL_Planning);
db1.COG_Sustained_Attention = double(db1.COG_Sustained_Attention);
%db1.COG_Spatial_Memory = double(db1.COG_Spatial_Memory);
%db1.COG_Strategy_SWM = double(db1.COG_Strategy_SWM);
db1.COG_SW_Memory = double(db1.COG_SW_Memory);
%db1.COG_Flex = double(db1.COG_Flex);
%db1.COG_Latency_CogFlex = double(db1.COG_Latency_CogFlex);
db1.COG_Reaction_Time = double(db1.COG_Reaction_Time);


%% Cognitive variable list
cognitiveVars = {
    'COG_DART'
 %   'COG_VIQ'
 %   'COG_PIQ'
    'COG_Verbal_Memory'
    'COG_Verbal_Wmemory'
    'COG_Verbal_Fluency'
    'COG_Processing_Speed'
    'COG_ToL_Planning'
    'COG_Sustained_Attention'
 %   'COG_Spatial_Memory'
 %   'COG_Strategy_SWM'
    'COG_SW_Memory'
 %   'COG_Flex'
 %   'COG_Latency_CogFlex'
    'COG_Reaction_Time'
};
%% Mean imputation
for v = 1:numel(cognitiveVars)
    x = db1.(cognitiveVars{v});
    db1.(cognitiveVars{v})(isnan(x)) = mean(x,'omitnan');
end

%% ROI data
roiVars = db1.Properties.VariableNames(contains(db1.Properties.VariableNames,'FD_'));
nROI = numel(roiVars);

dataArray = zeros(height(db1), nROI + numel(cognitiveVars));
dataArray(:,1:nROI) = table2array(db1(:,roiVars));
dataArray(:,nROI+1:end) = table2array(db1(:,cognitiveVars));

Long = array2table( ...
    dataArray, ...
    'VariableNames', [roiVars(:).' cognitiveVars(:).'] );

% Add covariates separately
Long.NU_Age = db1.NU_Age;
Long.NU_Sex = db1.NU_Sex;

%% Partial correlations
results = struct();

for i = 1:nROI
    roiData = Long{:,roiVars{i}};
    cognitiveData = Long{:,cognitiveVars};
    covariates = [Long.NU_Age, double(Long.NU_Sex)];

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

%% Significant Correlations Table Creation

% Initialize variables to store significant correlations
sigRois = {};
sigCogVars = {};
sigRhos  = [];
sigPvals = [];
sigQvals = [];

% Loop through all ROI and Cognitive variable pairs
for i = 1:nROI
    for j = 1:nCog
        if pMatrix(i, j) < alpha
            % Add significant correlation to the list
            sigRois{end+1, 1} = roiVars{i};       % ROI name
            sigCogVars{end+1, 1} = cognitiveVars{j}; % Cognitive variable name
            sigRhos(end+1, 1) = correlationMatrix(i, j); % Correlation coefficient (Rho)
            sigPvals(end+1, 1) = pMatrix(i, j);       % p-value
             sigQvals(end+1, 1)   = qMatrix(i, j);  %FDR-corrected q-value
        end
    end
end

sigCorrTable = table( ...
    sigRois, ...
    sigCogVars, ...
    sigRhos, ...
    sigPvals, ...
    sigQvals, ...
    'VariableNames', { ...
        'ROI', ...
        'Cognitive_Variable', ...
        'Partial_Correlation_Rho', ...
        'PValue', ...
        'QValue_FDR' ...
    });

% Display the table in the Command Window
disp('Significant Partial Correlations:');
disp(sigCorrTable);

% Save the table to a CSV file
writetable(sigCorrTable, 'Significant_Partial_Correlations.csv');
disp('Significant partial correlations saved to: Significant_Partial_Correlations.csv');



%% Scatterplots for all FDR-significant correlations (partial regression)

alpha_fdr = 0.05;

for i = 1:nROI
    for j = 1:nCog

        % Only plot FDR-significant correlations
        if qMatrix(i,j) < alpha_fdr

            roiName = roiVars{i};
            cogName = cognitiveVars{j};

            % Extract variables
            Y = Long.(roiName);        % ROI
            X = Long.(cogName);        % Cognitive measure
            C = [Long.NU_Age, double(Long.NU_Sex)];

            % Remove rows with NaNs
            validRows = ~any(isnan([Y X C]), 2);
            Y = Y(validRows);
            X = X(validRows);
            C = C(validRows,:);

            % Residualize with respect to covariates
            Y_res = Y - C * (C \ Y);
            X_res = X - C * (C \ X);

            % Partial correlation (for annotation)
            [rho, pval] = partialcorr(X, Y, C);

            % ===== Create figure =====
            figure; hold on;

            % Get group data for coloring
            g = db1.Group(validRows);

            % Scatter points colored by group
            gscatter(X_res, Y_res, g, [], [], 40, 'off');

            % Linear fit on residuals
            mdl = fitlm(X_res, Y_res);

            % Prediction range
            xFit = linspace(min(X_res), max(X_res), 100)';
            [yFit, yCI] = predict(mdl, xFit);

            % Confidence interval
            fill([xFit; flipud(xFit)], ...
                 [yCI(:,1); flipud(yCI(:,2))], ...
                 [0.7 0.7 0.7], ...
                 'FaceAlpha', 0.4, ...
                 'EdgeColor', 'none');

            % Regression line
            plot(xFit, yFit, 'k', 'LineWidth', 2);

            % Labels and title
            xlabel(strrep(cogName,'_','\_'), 'FontSize', 12);
            ylabel(strrep(roiName,'_','\_'), 'FontSize', 12);

            title(sprintf('%s vs %s (partial)', ...
                strrep(roiName,'_','\_'), ...
                strrep(cogName,'_','\_')));

            % Statistics annotation
            text(0.05, 0.95, ...
                sprintf('\\rho = %.2f\np = %.3g\nq = %.3g', ...
                rho, pval, qMatrix(i,j)), ...
                'Units','normalized', ...
                'VerticalAlignment','top', ...
                'FontSize', 11);

            box on;
            grid on;
            legend('show');

            hold off;

            % Optional: save figure
            filename = sprintf('Scatter_%s_%s.png', roiName, cogName);
            saveas(gcf, filename);

        end
    end
end
