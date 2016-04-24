function min_err_neuron = optimal_hneurons(X, T, maxNeurons)
%OPTIMIAL_HNEURONS Return the optmial number of hidden neurons
%   The number of neurons in a hidden layer is considered as a
%   hyper-parameter. The value of the hyper-paramenter is optmized using
%   10-fold cross-validation.
    min_err_neuron = 10;
    min_err = Inf;
    for hiddenNeurons = 10:maxNeurons
        CVO = cvpartition(T(:,1), 'k', 10);
        err = zeros(CVO.NumTestSets,1);
        for i = 1:CVO.NumTestSets
            trIdx = CVO.training(i);
            teIdx = CVO.test(i);
            train_inputs= X(trIdx,:);
            train_outputs= T(trIdx,:);
            test_inputs= X(teIdx,:);
            test_outputs= T(teIdx,:);
            net = patternnet(hiddenNeurons);
            net.trainParam.max_fail = 10;
            net.performParam.regularization = 0.01;
            net.trainParam.showWindow = false;
            net = train(net, train_inputs', train_outputs');
            y = net(test_inputs');
            [~,cm,~,~] = confusion(test_outputs', y);
            err(i) = 0.5*(cm(1,2)/(cm(1,1)+cm(1,2)) + cm(2,1)/(cm(2,1)+cm(2,2)));
        end
        all_err = sum(err)/CVO.NumTestSets;
        if all_err < min_err
            min_err = all_err;
            min_err_neuron = hiddenNeurons;
        end
    end
end

