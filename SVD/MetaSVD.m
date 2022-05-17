clear all; close all;
clc
%% LOAD DATA
load("elmo_all.mat");
load("unirep_all.mat");
load("transformer_all.mat");
d_elmo = size(elmo_data{1},2); % dimension of the elmo embedding
d_unirep = size(unirep_data{1},2); % dimension of the unirep embedding
d_transformer = size(transformer_data{1},2); % dimension of the transformer embedding
%% MERGE EACH DATASET
% Merge the train valid and test data for the ease of concatenation
elmo_all = [elmo_data{1};elmo_data{2};elmo_data{3};...
    elmo_data{4};elmo_data{5}];
unirep_all = [unirep_data{1};unirep_data{2};unirep_data{3};...
    unirep_data{4};unirep_data{5}];
transformer_all = [transformer_data{1};transformer_data{2};...
    transformer_data{3};transformer_data{4};transformer_data{5}];
%% NORMALIZE EACH EMBEDDING (via l2 normalization in each row)
 elmo_all = elmo_all./vecnorm(elmo_all')';
 unirep_all = unirep_all./vecnorm(unirep_all')';
 transformer_all = transformer_all./vecnorm(transformer_all')';
 concat_all = [elmo_all,unirep_all,transformer_all];
 % Data sizes
 n_all = size(elmo_all,1);
 n_train = size(elmo_data{1},1);
 n_valid = size(elmo_data{2},1);
 n_family = size(elmo_data{3},1);
 n_superfamily = size(elmo_data{4},1);
 n_fold = size(elmo_data{5},1);
 % Split the concatenated data into train valid test_family test_superfamily test_fold
 concat_train = concat_all(1:n_train,:);
 concat_valid = concat_all(n_train+1:n_train+n_valid,:);
 concat_test_family = concat_all(n_train+n_valid+1:n_train+n_valid+n_family,:);
 concat_test_superfamily = concat_all(n_train+n_valid+n_family+1:n_train+n_valid+n_family+n_superfamily,:);
 concat_test_fold = concat_all(n_train+n_valid+n_family+n_superfamily+1:end,:);
%% SVD META-EMBEDDINGS
d_svd = 1000:250:3500;
% We take the truncated SVD
[U_train, S_train, V_train] = svd(concat_train,0);
S_inv = inv(S_train);
% Generate the valid test_family test_superfamily test_fold data
U_valid = concat_valid*V_train*S_inv;
U_test_family = concat_test_family*V_train*S_inv;
U_test_superfamily = concat_test_superfamily*V_train*S_inv;
U_test_fold = concat_test_fold*V_train*S_inv;
% We only take the first d columns of the U matrices
 for i = 1:numel(d_svd)
     d = d_svd(i);
     meta_train = U_train(:,1:d);
     meta_valid = U_valid(:,1:d);
     meta_test_family = U_test_family(:,1:d);
     meta_test_superfamily = U_test_superfamily(:,1:d);
     meta_test_fold = U_test_fold(:,1:d);
     % Save data to .mat file
     fn = "meta_svd_"+num2str(d);
     save(fn, 'meta_train', 'meta_valid', 'meta_test_family',...
         'meta_test_superfamily', 'meta_test_fold');
 end
