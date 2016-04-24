arcene_train_data   = load('arcene_train_data');
arcene_train_labels = load('arcene_train_labels');
arcene_valid_data   = load('arcene_valid_data');
arcene_valid_labels = load('arcene_valid_labels');

arcene_train_labels(arcene_train_labels==-1) = 0;
arcene_valid_labels(arcene_valid_labels==-1) = 0;

% FEATURE SELECTION
[ranked,~] = relieff(arcene_train_data, arcene_train_labels, 10);

arcene_train_sub = arcene_train_data(:,ranked(1:150));
arcene_valid_sub = arcene_valid_data(:,ranked(1:150));

%{
% STANDARDIZE DATA

% get mean and sd of each feature in the input data
mean_train = mean(arcene_train_sub_norm);
sd_train   = std(arcene_train_sub_norm);

arcene_train_sub_norm = bsxfun(@rdivide, bsxfun(@minus, arcene_train_sub_norm, mean_train), sd_train);
arcene_valid_sub_norm = bsxfun(@rdivide, bsxfun(@minus, arcene_valid_sub_norm, mean_train), sd_train);
%}

% NORMALIZE DATA

% get min and max of each feature in the input data
min_train = min(arcene_train_sub);
max_train   = max(arcene_train_sub);
min_max_diff = bsxfun(@minus, max_train, min_train);

arcene_train_sub_norm = bsxfun(@rdivide, bsxfun(@minus, arcene_train_sub, min_train), min_max_diff);
arcene_valid_sub_norm = bsxfun(@rdivide, bsxfun(@minus, arcene_valid_sub, min_train), min_max_diff);

% TRAIN
rng('default');
nnet = patternnet(24,'trainbr','sse');
% nnet.trainParam.showWindow = false;

[nnet, tr] = train(nnet, arcene_train_sub_norm', arcene_train_labels');

predictions = nnet(arcene_valid_sub_norm');

plotconfusion(arcene_valid_labels', predictions);
[~,cm,~,~] = confusion(arcene_valid_labels', predictions);
disp(0.5*(cm(1,2)/(cm(1,1)+cm(1,2)) + cm(2,1)/(cm(2,1)+cm(2,2))));
