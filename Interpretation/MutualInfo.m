% This code computes the Mutual Information between embeddings, under joint normality assumptions.
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
[R_elmo, P_elmo] = corrcoef(elmo);
[R_unirep, P_unirep] = corrcoef(unirep);
[R_transformer, P_transformer] = corrcoef(transformer);
%% COVARIANCE MATRICES AMONG FEATURES
C_e = cov(elmo);
C_u = cov(unirep);
C_t = cov(transformer);
%% COVARIANCE MATRICES AMONG EMBEDDING PAIRS
C_et = cov([elmo,transformer]);
C_eu = cov([elmo,unirep]);
C_ut = cov([unirep,transformer]);
%% DETERMINANTS OF COVARIANCE MATRICES
% For numeric stability we look at the log of singular values to compute
% determinants.
[~,S_e,~]=svd(C_e);
[~,S_u,~]=svd(C_u);
[~,S_t,~]=svd(C_t);
[~,S_et,~]=svd(C_et);
[~,S_eu,~]=svd(C_eu);
[~,S_ut,~]=svd(C_ut);
det_log_elmo = sum(log2(diag(S_e)));
det_log_unirep = sum(log2(diag(S_u)));
det_log_transformer = sum(log2(diag(S_t)));
det_log_et = sum(log2(diag(S_et)));
det_log_eu = sum(log2(diag(S_eu)));
det_log_ut = sum(log2(diag(S_ut)));
%% MUTUAL INFORMATION OF JOINTLY GAUSSIANS
% Mutual Information between two jointly gaussian distributions is 
% MI = 1/2 log2(det(Cov_1) det(Cov_2)/det(Cov_joint))
% For numeric stability we work in the log domain directly.
% I(Elmo;Transformer)
MI_et = 0.5*(det_log_elmo+det_log_transformer-det_log_et);
% I(Elmo;Unirep)
MI_eu = 0.5*(det_log_elmo+det_log_unirep-det_log_eu);
% I(Unirep;Transformer)
MI_ut = 0.5*(det_log_unirep+det_log_transformer-det_log_ut);