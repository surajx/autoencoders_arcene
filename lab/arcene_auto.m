arcene_train_data   = load('arcene_train_data');
arcene_train_labels = load('arcene_train_labels');
arcene_valid_data   = load('arcene_valid_data');
arcene_valid_labels = load('arcene_valid_labels');

arcene_train_labels(arcene_train_labels==-1) = 0;
arcene_valid_labels(arcene_valid_labels==-1) = 0;

X = arcene_train_data';
Y = arcene_train_labels';

hiddenSize = 10;
autoenc1 = trainAutoencoder(X,hiddenSize,...
    'L2WeightRegularization',0.001,...
    'SparsityRegularization',4,...
    'SparsityProportion',0.05,...
    'DecoderTransferFunction','purelin');

features1 = encode(autoenc1,X);

autoenc2 = trainAutoencoder(features1,hiddenSize,...
    'L2WeightRegularization',0.001,...
    'SparsityRegularization',4,...
    'SparsityProportion',0.05,...
    'DecoderTransferFunction','purelin',...
    'ScaleData',false);

features2 = encode(autoenc2,features1);
softnet = trainSoftmaxLayer(features2,Y,'LossFunction','crossentropy');

deepnet = stack(autoenc1,autoenc2, softnet);

deepnet = train(deepnet, X, Y);

predictions = deepnet(arcene_valid_data');
%plotconfusion(arcene_valid_labels', predictions);
[c,cm,ind,per] = confusion(arcene_valid_labels', predictions);
disp(c);