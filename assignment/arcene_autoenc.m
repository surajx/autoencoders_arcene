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

% get min and max of each feature in the input data
min_train = min(arcene_train_sub);
max_train   = max(arcene_train_sub);
min_max_diff = bsxfun(@minus, max_train, min_train);

arcene_train_sub_norm = bsxfun(@rdivide, bsxfun(@minus, arcene_train_sub, min_train), min_max_diff);
arcene_valid_sub_norm = bsxfun(@rdivide, bsxfun(@minus, arcene_valid_sub, min_train), min_max_diff);


X = arcene_train_sub_norm';
Y = arcene_train_labels';

rng('default');
hiddenSize = 50;
autoenc1 = trainAutoencoder(X,hiddenSize,...
    'L2WeightRegularization',10,...
    'DecoderTransferFunction','purelin');

features1 = encode(autoenc1,X);

autoenc2 = trainAutoencoder(features1,hiddenSize,...
    'L2WeightRegularization',10,...
    'DecoderTransferFunction','purelin',...
    'ScaleData',false);

features2 = encode(autoenc2,features1);
softnet = trainSoftmaxLayer(features2,Y,'LossFunction','crossentropy');

deepnet = stack(autoenc1,autoenc2, softnet);

deepnet = train(deepnet, X, Y);

predictions = deepnet(arcene_valid_sub_norm');
plotconfusion(arcene_valid_labels', predictions);
[~,cm,~,~] = confusion(arcene_valid_labels', predictions);
disp(0.5*(cm(1,2)/(cm(1,1)+cm(1,2)) + cm(2,1)/(cm(2,1)+cm(2,2))));
