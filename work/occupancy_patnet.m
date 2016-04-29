occ_data = csvread('occupancy_train',1,2);
occ_train = occ_data(:,2:end-1);
occ_train_label = occ_data(:,end);

occ_data = csvread('occupancy_test_1',1,2);
occ_test_1 = occ_data(:,2:end-1);
occ_test_1_label = occ_data(:,end);


occ_train_new = cat(1,occ_train,occ_test_1);
occ_train_label_new = cat(1, occ_train_label, occ_test_1_label);

occ_data = csvread('occupancy_test_2',1,2);
occ_test_2 = occ_data(:,2:end-1);
occ_test_2_label = occ_data(:,end);

CVO = cvpartition(occ_train_label_new(:,1), 'k', 10);

err = zeros(CVO.NumTestSets,1);
for i = 1:CVO.NumTestSets
    trIdx = CVO.training(i);
    teIdx = CVO.test(i);
    train_inputs= occ_train_new(trIdx,:);
    train_outputs= occ_train_label_new(trIdx,:);
    test_inputs= occ_train_new(teIdx,:);
    test_outputs= occ_train_label_new(teIdx,:);
    net = patternnet(10);
    net = train(net, train_inputs', train_outputs');
    y = net(test_inputs');
    %err(i) = sum(round(y)~=test_outputs')/length(test_outputs');
    [c,cm,ind,per] = confusion(test_outputs', y);
    err(i) = c;
end

all_err = sum(err)/CVO.NumTestSets;
disp(all_err)
predictions = net(occ_test_2');
plotconfusion(occ_test_2_label', predictions);
%nnet = patternnet(100);

%nnet.divideParam.trainRatio = 0.8;
%nnet.divideParam.valRatio = 0.10;
%nnet.divideParam.testRatio = 0.10;
%nnet.trainParam.showWindow = false;

%err_sum = 0;
%for i = 1:50
    %[nnet, tr] = train(nnet, occ_train', occ_train_label');

    %predictions = nnet(occ_test_1');
    %plotconfusion(occ_test_1_label', predictions);

    %predictions = nnet(occ_test_2');
    %plotconfusion(occ_test_2_label', predictions);
        
    %[c,cm,ind,per] = confusion(arcene_valid_labels', predictions);
    %err_sum = err_sum + c;
%end
%disp(err_sum/50);