arcene_train_data   = load('arcene_train_data');
arcene_train_labels = load('arcene_train_labels');
arcene_valid_data   = load('arcene_valid_data');
arcene_valid_labels = load('arcene_valid_labels');

arcene_train_labels(arcene_train_labels==-1) = 0;
arcene_valid_labels(arcene_valid_labels==-1) = 0;


% FEATURE SELECTION
arcene_train_sub = arcene_train_data(:,var(arcene_train_data)~=0);
arcene_valid_sub = arcene_valid_data(:,var(arcene_train_data)~=0);

% NORMALIZE DATA

% get mean and sd of each feature in the input data
mean_train = mean(arcene_train_sub);
sd_train   = std(arcene_train_sub);

arcene_train_sub_norm = bsxfun(@rdivide, bsxfun(@minus, arcene_train_sub, mean_train), sd_train);
arcene_valid_sub_norm = bsxfun(@rdivide, bsxfun(@minus, arcene_valid_sub, mean_train), sd_train);

mask_corr = [];
for col = 1:size(arcene_train_sub_norm,2)
    R = corrcoef(arcene_train_sub_norm(:,col), arcene_train_labels);
    if (R(1,2)>=-0.1) && (R(1,2)<=0.1)
        mask_corr = [mask_corr col];
    end
end
arcene_train_sub_norm(:,mask_corr) = [];
arcene_valid_sub_norm(:,mask_corr) = [];

min_err_neuron = 10;
min_err = Inf;
for hiddenNeurons = 10:100
    CVO = cvpartition(arcene_train_labels(:,1), 'k', 10);
    err = zeros(CVO.NumTestSets,1);
    for i = 1:CVO.NumTestSets
        trIdx = CVO.training(i);
        teIdx = CVO.test(i);
        train_inputs= arcene_train_sub_norm(trIdx,:);
        train_outputs= arcene_train_labels(trIdx,:);
        test_inputs= arcene_train_sub_norm(teIdx,:);
        test_outputs= arcene_train_labels(teIdx,:);
        net = patternnet(hiddenNeurons);
        net.trainParam.showWindow = false;
        net = train(net, train_inputs', train_outputs');
        y = net(test_inputs');
        %err(i) = sum(round(y)~=test_outputs')/length(test_outputs');
        [c,~,~,~] = confusion(test_outputs', y);
        err(i) = c;
    end
    all_err = sum(err)/CVO.NumTestSets;
    if all_err < min_err
        min_err = all_err;
        min_err_neuron = hiddenNeurons;
    end
end

nnet = patternnet(min_err_neuron);

nnet.divideParam.trainRatio = 0.8;
nnet.divideParam.valRatio = 0.10;
nnet.divideParam.testRatio = 0.10;
nnet.trainParam.showWindow = false;

[nnet, tr] = train(nnet, arcene_train_sub_norm', arcene_train_labels');

predictions = nnet(arcene_valid_sub_norm');
%plotconfusion(arcene_valid_labels', predictions);
[c,cm,ind,per] = confusion(arcene_valid_labels', predictions);
disp(c);
