function min_err_net = generalized_nnet(X, T, H)
%GENERALIZED NEURAL NETWORK Return a generalized neural network for a given
%value of hidden neurons. The NN is generalized using 10-fold
%cross-validation.

    %container for the 10 NNs trained from the 10-fold validation
    NN = cell(1,10);
    %Initialize and configure the NN
    % Training Fn: Scaled Conjugate Gradient Back Propagation
    % Performance Fn: Cross-Entropy
    % Cross-validation Max Fail: The training process is cross-validated 
    % to stop early if the validation error rises for 10 consecutive 
    % iterations.
    % Regularization: To prevent over-fitting
    net = patternnet(H);
	net.divideParam.trainRatio = 0.75;
	net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.15;
    net.trainParam.max_fail = 10;
    net.performParam.regularization = 0.01;
	net.trainParam.showWindow = false;
    min_err = Inf;
    % Re-training and cross-validation.
    CVO = cvpartition(T(:,1), 'k', 10);
    for i = 1:CVO.NumTestSets
        trIdx = CVO.training(i);
        teIdx = CVO.test(i);
        train_inputs= X(trIdx,:);
        train_outputs= T(trIdx,:);
        test_inputs= X(teIdx,:);
        test_outputs= T(teIdx,:);
        NN{i} = train(net, train_inputs', train_outputs');
        y = NN{i}(test_inputs');
        [~,cm,~,~] = confusion(test_outputs', y);
        err = 0.5*(cm(1,2)/(cm(1,1)+cm(1,2)) + cm(2,1)/(cm(2,1)+cm(2,2)));
        if err < min_err
            min_err = err;
            min_err_net = NN{i};
        end
    end
end

