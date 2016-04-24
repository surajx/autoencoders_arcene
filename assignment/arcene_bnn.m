arcene_train_data   = load('arcene_train_data');
arcene_train_labels = load('arcene_train_labels');
arcene_valid_data   = load('arcene_valid_data');
arcene_valid_labels = load('arcene_valid_labels');

arcene_train_labels(arcene_train_labels==-1) = 0;
arcene_valid_labels(arcene_valid_labels==-1) = 0;


% FEATURE SELECTION
arcene_train_sub = arcene_train_data(:,var(arcene_train_data)~=0);
arcene_valid_sub = arcene_valid_data(:,var(arcene_train_data)~=0);

arcene_train_sub_norm = arcene_train_sub;
arcene_valid_sub_norm = arcene_valid_sub;

mask_corr = [];
for col = 1:size(arcene_train_sub_norm,2)
    R = corrcoef(arcene_train_sub_norm(:,col), arcene_train_labels);
    if (R(1,2)>=-0.1) && (R(1,2)<=0.1)
        mask_corr = [mask_corr col];
    end
end
arcene_train_sub_norm(:,mask_corr) = [];
arcene_valid_sub_norm(:,mask_corr) = [];

mask_SNR = [];
for col = 1:size(arcene_train_sub_norm,2)
    mean_diff = mean(arcene_train_sub_norm(:,col)) - mean(arcene_train_labels);
    var_sum = std(arcene_train_sub_norm(:,col)) + std(arcene_train_labels);
    SNR = mean_diff/var_sum;
    mask_SNR = [mask_SNR; [SNR col]];
end

mask_SNR = sortrows(mask_SNR);
mask_SNR(1:size(mask_SNR)-125,:) = [];

arcene_train_sub_norm = arcene_train_sub_norm(:,mask_SNR(:,2)');
arcene_valid_sub_norm = arcene_valid_sub_norm(:,mask_SNR(:,2)');


% NORMALIZE DATA

% get mean and sd of each feature in the input data
mean_train = mean(arcene_train_sub_norm);
sd_train   = std(arcene_train_sub_norm);

arcene_train_sub_norm = bsxfun(@rdivide, bsxfun(@minus, arcene_train_sub_norm, mean_train), sd_train);
arcene_valid_sub_norm = bsxfun(@rdivide, bsxfun(@minus, arcene_valid_sub_norm, mean_train), sd_train);

% TRAIN
rng('default');
nnet = patternnet(20,'trainbr','sse');
% nnet.trainParam.showWindow = false;

[nnet, tr] = train(nnet, arcene_train_sub_norm', arcene_train_labels');

predictions = nnet(arcene_valid_sub_norm');

plotconfusion(arcene_valid_labels', predictions);
[~,cm,~,~] = confusion(arcene_valid_labels', predictions);
disp(0.5*(cm(1,2)/(cm(1,1)+cm(1,2)) + cm(2,1)/(cm(2,1)+cm(2,2))));
