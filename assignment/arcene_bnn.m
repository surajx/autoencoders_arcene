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
arcene_train_sub = arcene_train_sub(:,ranked(1:150));
arcene_valid_sub = arcene_valid_sub(:,ranked(1:150));


% STANDARDIZE DATA

% get mean and sd of each feature in the input data
mean_train = mean(arcene_train_sub);
sd_train   = std(arcene_train_sub);

arcene_train_sub_norm = bsxfun(@rdivide, bsxfun(@minus, arcene_train_sub, mean_train), sd_train);
arcene_valid_sub_norm = bsxfun(@rdivide, bsxfun(@minus, arcene_valid_sub, mean_train), sd_train);

% TRAIN
rng('default');
nnet = patternnet(24,'trainbr','sse');
% nnet.trainParam.showWindow = false;

[nnet, tr] = train(nnet, arcene_train_sub_norm', arcene_train_labels');

predictions = nnet(arcene_valid_sub_norm');

plotconfusion(arcene_valid_labels', predictions);
[~,cm,~,~] = confusion(arcene_valid_labels', predictions);
disp(0.5*(cm(1,2)/(cm(1,1)+cm(1,2)) + cm(2,1)/(cm(2,1)+cm(2,2))));
