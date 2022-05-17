clear all;
%% LOAD DATA
load("elmo_all.mat");
load("unirep_all.mat");
load("transformer_all.mat");
d_elmo = size(elmo_data{1},2); % dimension of the elmo embedding
d_unirep = size(unirep_data{1},2); % dimension of the unirep embedding
d_transformer = size(transformer_data{1},2); % dimension of the transformer embedding
%% TRAINING DATASETS
elmo = elmo_data{1};
unirep = unirep_data{1};
transformer = transformer_data{1};
%% NORMALIZE EACH EMBEDDING
elmo = elmo./vecnorm(elmo')';
unirep = unirep./vecnorm(unirep')';
transformer = transformer./vecnorm(transformer')';
%% GRADIENT DESCENT FOR 1toN
d_vec = [d_elmo d_unirep d_transformer]; % dimensions of embeddings
n = size(elmo,1); % sample size
k = numel(d_vec); % number of embeddings
d = 1000; % HYPERPARAM: dimension of meta-embedding 
lambda=1; % HYPERPARAM: regularization factor
% Embeddings
W = cell(1,k);
W{1} = elmo;
W{2} = unirep;
W{3} = transformer;
%% GRADIENT DESCENT INITIALIZATION
alpha = 5e-5; % learning rate
thresh = 10; % threshold for iteration
useSavedWeights = false; % start from scratch
% first iteration
gradmat = @(w,i,M,W,lambda) 2*(w*(w.'*M{i}.')+lambda*M{i}.'-w*W{i});
M = cell(1,k);
if useSavedWeights
    % We used the saved weigths as seeds
    load('bestRes_noTest_d2250.mat')
else
    % Initializations
    for i=1:k
        M{i} = eye(d_vec(i),d);
    end
    % Take the first d dimensions of the concatenations as
    % the initialization of w
    ww = [W{1},W{2},W{3}];
    w = ww(:,1:d).';
    % or initialize randomly
    % w = rand(d,n);
    % partial derivative of J wrt M_i
    % First iteration
    M_prev = M;
    w_prev = w;
    for i = 1:k
        M{i} = M_prev{i} - alpha*2*gradmat(w,i,M_prev,W,lambda).';
    end
    w = w_prev - alpha*2*gradw(w_prev,W,M_prev);
    iter = 1;
end
% the cost function
J = obJ(M,W,w,lambda);
sprintf('Iteration: %d    Loss:%f',iter,J)
%% GRADIENT DESCENT
while J>thresh
    M_prev = M;
    w_prev = w;
    J_prev = J;
    % Update M_i
    for i = 1:k
        M{i} = M_prev{i} - alpha*2*gradmat(w,i,M_prev,W,lambda).';
    end
    % Update w
    w = w_prev - alpha*2*gradw(w_prev,W,M_prev);
    iter = iter+1;
    % New cost
    J=obJ(M,W,w,lambda);
    % Displaying the convergence
    sprintf('Iteration: %d    Loss:%f',iter,J)
    % could also display the convergence of |M_i w-w_i|
    %[norm(M{1}*w-W{1}.'),norm(M{2}*w-W{2}.'),norm(M{3}*w-W{3}.'),norm(M{4}*w-W{4}.')]
    % if t+1st iteration does not improve, go back and reduce learning rate
    if J_prev < J
        iter_last = iter-1;
        J = J_prev;
        w = w_prev;
        M = M_prev;
        iter = iter-1;
        alpha = alpha*0.8;
        disp('LEARNING RATE DROPPED!')
    end
    save bestRes_noTest_d2250.mat J w M iter alpha
end
%% GENERATING THE 3-LEVEL TEST DATASETS
elmo_train = (pinv(M{1})*elmo_data{1}')';
elmo_valid = (pinv(M{1})*elmo_data{2}')';
elmo_test_family = (pinv(M{1})*elmo_data{3}')';
elmo_test_superfamily = (pinv(M{1})*elmo_data{4}')';
elmo_test_fold = (pinv(M{1})*elmo_data{5}')';
unirep_train = (pinv(M{2})*unirep_data{1}')';
unirep_valid = (pinv(M{2})*unirep_data{2}')';
unirep_test_family = (pinv(M{2})*unirep_data{3}')';
unirep_test_superfamily = (pinv(M{2})*unirep_data{4}')';
unirep_test_fold = (pinv(M{2})*unirep_data{5}')';
transformer_train = (pinv(M{3})*transformer_data{1}')';
transformer_valid = (pinv(M{3})*transformer_data{2}')';
transformer_test_family = (pinv(M{3})*transformer_data{3}')';
transformer_test_superfamily = (pinv(M{3})*transformer_data{4}')';
transformer_test_fold = (pinv(M{3})*transformer_data{5}')';
% Each of these should approximate the same set of meta-embeddings.
% We take the average to get the meta-embeddings
meta_train = (elmo_train+unirep_train+transformer_train)/3;
meta_valid = (elmo_valid+unirep_valid+transformer_valid)/3;
meta_test_family = (elmo_test_family+unirep_test_family+transformer_test_family)/3;
meta_test_superfamily = (elmo_test_superfamily+unirep_test_superfamily+transformer_test_superfamily)/3;
meta_test_fold = (elmo_test_fold+unirep_test_fold+transformer_test_fold)/3;

%% FUNCTIONS
function [ gw ] = gradw(w,W,M )
%   compute the gradient of w
k = length(W);
gw = 0;
for i=1:k
    gw = gw + M{i}.'*(M{i}*w)-M{i}.'*W{i}.';
end
gw = 2* gw;
end
function [J] = obJ(M,W,w,lambda)
    % computes the cost function
    k = length(M);
    sum = 0;
    for i=1:k
        sum = sum+norm(M{i}*w-W{i}.')^2+lambda*norm(M{i},'fro')^2;
    end
    J = sum;
end
