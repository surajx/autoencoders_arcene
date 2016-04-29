occ_data = csvread('default_cc.csv',2,1);
occ_X = occ_data(:,2:end-1);
occ_T = occ_data(:,end);

occ_train = occ_X(1:int32(floor(0.75 * end)),:);
occ_train_label = occ_T(1:int32(floor(0.75 * end)),:);

occ_test  = occ_X(int32(floor(0.75 * end))+1:end,:);
occ_test_label  = occ_T(int32(floor(0.75 * end))+1:end,:);


nnet = patternnet(1000);

nnet.divideParam.trainRatio = 0.70;
nnet.divideParam.valRatio = 0.15;
nnet.divideParam.testRatio = 0.15;
%nnet.trainParam.showWindow = false;

[nnet, tr] = train(nnet, occ_train', occ_train_label');

predictions = nnet(occ_test');
plotconfusion(occ_test_label', predictions);
%[c,cm,ind,per] = confusion(arcene_valid_labels', predictions);