% This code computes and plots the histogram of the distribution of the number of strongly-correlated features for a given feature in the concatenated dataset.
%% LOAD DATA
load("elmo_all.mat");
load("unirep_all.mat");
load("transformer_all.mat");
d_elmo = size(elmo_data{1},2); % dimension of the elmo embedding
d_unirep = size(unirep_data{1},2); % dimension of the unirep embedding
d_transformer = size(transformer_data{1},2); % dimension of the transformer embedding
%% MERGE EACH DATASET
% We need each data point to improve our estimation. We use all 5 datasets.
elmo = [elmo_data{1};elmo_data{2};elmo_data{3};...
    elmo_data{4};elmo_data{5}];
unirep = [unirep_data{1};unirep_data{2};unirep_data{3};...
    unirep_data{4};unirep_data{5}];
transformer = [transformer_data{1};transformer_data{2};...
    transformer_data{3};transformer_data{4};transformer_data{5}];
%% NORMALIZE EACH EMBEDDING (via l2 normalization in each row)
 elmo = elmo./vecnorm(elmo')';
 unirep = unirep./vecnorm(unirep')';
 transformer = transformer./vecnorm(transformer')';
%% CORRELATION COEFFICIENTS AMONG FEATURES
% [R,P] returns the matrices R storing the correlation coefficients between
% features and P storing the p-value of the corresponding correlation.
% A p-value smaller than .05 indicates strong correlation.
[R, P] = corrcoef([elmo,transformer,unirep]);
% Matrix S stores the indicator functions regarding strong correlation
S = P<0.05;
% The sum of columns in S in row i gives the number of features strongly
% correlated with the i-th feature
mu = mean(sum(S,2)); % mean value
sigma = std(sum(S,2)); % standard deviation
% We now plot the histogram of the distribution of the number of features
% strongly correlated with for any given feature.
figure
histogram(sum(S,2))