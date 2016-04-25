arcene_train_data   = load('arcene_train_data');
arcene_train_labels = load('arcene_train_labels');
arcene_valid_data   = load('arcene_valid_data');
arcene_valid_labels = load('arcene_valid_labels');

%Convert -1 to 0 for Classification problems.
arcene_train_labels(arcene_train_labels==-1) = 0;
arcene_valid_labels(arcene_valid_labels==-1) = 0;

% FEATURE SELECTION
% Remove feature with zero variance.
arcene_train_sub = arcene_train_data(:,var(arcene_train_data)~=0);
arcene_valid_sub = arcene_valid_data(:,var(arcene_train_data)~=0);

% Remove feature with correlation with output lying in range [-0.1,0.1]
mask_corr = [];
for col = 1:size(arcene_train_sub,2)
    R = corrcoef(arcene_train_sub(:,col), arcene_train_labels);
    if (R(1,2)>=-0.1) && (R(1,2)<=0.1)
        mask_corr = [mask_corr col];
    end
end
arcene_train_sub(:,mask_corr) = [];
arcene_valid_sub(:,mask_corr) = [];

% Use RELIEFF Ranking to score features and select top 1400 ranked
% features as final feature subset.
[ranked,~] = relieff(arcene_train_sub, arcene_train_labels, 10);
arcene_train_sub = arcene_train_sub(:,ranked(1:1400));
arcene_valid_sub = arcene_valid_sub(:,ranked(1:1400));

% NORMALIZE DATA
% get min and max of each feature in the input data
min_train = min(arcene_train_sub);
max_train   = max(arcene_train_sub);
min_max_diff = bsxfun(@minus, max_train, min_train);

arcene_train_sub_norm = bsxfun(@rdivide, bsxfun(@minus, arcene_train_sub, min_train), min_max_diff);
arcene_valid_sub_norm = bsxfun(@rdivide, bsxfun(@minus, arcene_valid_sub, min_train), min_max_diff);

% Setting seed of random number generator so that subsequent runs of the
% neural network produce the same result.
rng('default');
min_err = Inf;
opt_neuron = 1;

% Optimize the number of neurons as a hyper parameter and simultaneously,
% generalize the neural network to training data with 10-fold cross
% validation.
for H = 1:100
    nnet = generalized_nnet(arcene_train_sub_norm, arcene_train_labels, H);
    predictions = nnet(arcene_valid_sub_norm');
    [~,cm,~,~] = confusion(arcene_valid_labels', predictions);
    err = 0.5*(cm(1,2)/(cm(1,1)+cm(1,2)) + cm(2,1)/(cm(2,1)+cm(2,2)));
    if err < min_err
        min_err = err;
        opt_neuron = H;
        min_err_net = nnet;
    end
end
predictions = min_err_net(arcene_valid_sub_norm');
plotconfusion(arcene_valid_labels', predictions);
disp(opt_neuron);
disp(min_err);
disp(min_err_net);
