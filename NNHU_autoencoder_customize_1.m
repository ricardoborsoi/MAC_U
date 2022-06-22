function [Hest,W,MLP,rmse] = NNHU_autoencoder_customize_1(Y,M_prior,lambda1,...
lambda2,lambda3,lambda5,lambda6,learnRate,numEpochs,rngflag)
% Y: image, (row*bands)by col
% c: number of endmembers (not used anymore)
% H: abundances
% W: endmembers
if (rngflag)
    rng(10, 'twister') % reproducible
end
W=M_prior;
Co_H=inv(W'*W)*W';
H=inv(W'*W)*W'*Y;
size_Y=size(Y,1);
size_H=size(H,1);
size_M=size_Y*size_H+size_H;
miniBatchSize  = 128;
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.95;
input=Y;
inputsize=size(input,1);
% output=Y';
trailingAvgMLP = [];
trailingAvgSqMLP = [];
Weight_concat=[eye(size_H);zeros(size_Y*size_H,size_H)];
Bias_concat=[zeros(size_H,1);W(:)];
stop_criterion = 1e-3;
loss_0 = 1;
loss = 1;
past_loss = 0;
% Model Build
init_factor = 1e-5;
layers = [
    featureInputLayer(inputsize,'Name','input','Normalization','none')  %featureinputlayer  comment normalization
    fullyConnectedLayer(inputsize*2,'Name','fc_11')   % 2 layers   initialization with small value instead of 0.
    leakyReluLayer('Name','relu_11')
    fullyConnectedLayer(round(inputsize/2),'Name','fc_1')   
    leakyReluLayer('Name','relu_1')
    fullyConnectedLayer(round(inputsize/8),'Name','fc_2')   
    leakyReluLayer('Name','relu_2')
    fullyConnectedLayer(12,'Name','fc_3')  
    leakyReluLayer('Name','relu_3')
    fullyConnectedLayer(size_H,'Name','fc_4')
    leakyReluLayer('Name','relu_4')
    fullyConnectedLayer(size_H,'Name','ratio_1','Weights',init_factor *eye(size_H),'Bias',1e-40.*ones(size_H,1),'BiasLearnRateFactor',0)
    additionLayer(2,'Name','add_1')
    fullyConnectedLayer(size_H,'Name','fc_add_1','Weights',eye(size_H),'Bias',1e-40.*ones(size_H,1),'WeightLearnRateFactor',0,'BiasLearnRateFactor',0)
%     softmaxLayer('Name','softmax_1')
    UntilityLayer('softmax_1')
    
    fullyConnectedLayer(size_M,'Name','concatenate','Weights',Weight_concat,'Bias',Bias_concat,'WeightLearnRateFactor',0,'BiasLearnRateFactor',0)
    fullyConnectedLayer(size_H*size_Y,'Name','fc_5')
    leakyReluLayer('Name','relu_5')
    fullyConnectedLayer(size_Y,'Name','fc_6')
    leakyReluLayer('Name','relu_6')
    fullyConnectedLayer(size_Y,'Name','fc_7')
    leakyReluLayer('Name','relu_7')
    fullyConnectedLayer(size_Y,'Name','ratio_2','Weights',init_factor.*ones(size_Y,size_Y),'Bias',1e-40.*ones(size_Y,1),'BiasLearnRateFactor',0)
    additionLayer(2,'Name','add_2')
    fullyConnectedLayer(size_Y,'Name','fc_add_2','Weights',eye(size_Y),'Bias',1e-40.*ones(size_Y,1),'WeightLearnRateFactor',0,'BiasLearnRateFactor',0)
    ];

lgraph = layerGraph(layers);
skipfc1 = fullyConnectedLayer(size_H,'Name','skipfc_1','Weights',Co_H,'Bias',1e-40.*ones(size_H,1),'BiasLearnRateFactor',0);
ratio1=fullyConnectedLayer(size_H,'Name','skipratio_1','Weights',eye(size_H)+1e-40.*ones(size_H,1),'Bias',1e-40.*ones(size_H,1),'BiasLearnRateFactor',0);
lgraph = addLayers(lgraph,skipfc1);
lgraph = addLayers(lgraph,ratio1);
lgraph = connectLayers(lgraph,'input','skipfc_1');
lgraph = connectLayers(lgraph,'skipfc_1','skipratio_1');
lgraph = connectLayers(lgraph,'skipratio_1','add_1/in2');

skipfc2 = fullyConnectedLayer(size_Y,'Name','skipfc_2','Weights',W,'Bias',1e-40.*ones(size_Y,1));
skipfc2.BiasLearnRateFactor=0;
ratio2=fullyConnectedLayer(size_Y,'Name','skipratio_2','Weights',eye(size_Y),'Bias',1e-40.*ones(size_Y,1));
ratio2.BiasLearnRateFactor=0;
lgraph = addLayers(lgraph,skipfc2);
lgraph = addLayers(lgraph,ratio2);
lgraph = connectLayers(lgraph,'softmax_1','skipfc_2');
lgraph = connectLayers(lgraph,'skipfc_2','skipratio_2');
lgraph = connectLayers(lgraph,'skipratio_2','add_2/in2');
% figure
% plot(lgraph)
MLP = dlnetwork(lgraph);
inputdatastore=MySequenceDatastore(input');
inputdatastore.MiniBatchSize=miniBatchSize;
iteration = 0;
% figure;
% h=animatedline('Color',[0 0.447 0.741]);
% start = tic;
for epoch = 1:numEpochs
    if mod(epoch,3)==0
        disp(['Epoch = ', num2str(epoch), ', StopCriterion = ', num2str(c)]);
    end
    reset(inputdatastore);
    % Reset and shuffle datastore.
    shuffle(inputdatastore);
%     if(~rem(epoch,1))
%         learnRate=learnRate*(1-LearnrateDecayfactor);
%     end
    % Loop over mini-batches.
    while hasdata(inputdatastore)
        iteration = iteration + 1;
        
        % Read mini-batch of data.
        dlX = read(inputdatastore);
        
        % Generate latent inputs for the generator network. Convert to
        % dlarray and specify the dimension labels 'SSCB' (spatial,
        % spatial, channel, batch). If training on a GPU, then convert
        % latent inputs to gpuArray.      
        
        
        % Evaluate the model gradients and the generator state using
        % dlfeval and the modelGradients function listed at the end of the
        % example.
        [gradientvalue, lossvalue] = ...
            dlfeval(@modelGradients, MLP, dlX,lambda1,lambda2,lambda3,lambda5,lambda6,M_prior);                
        
        % Update the discriminator network parameters.
        [MLP,trailingAvgMLP,trailingAvgSqMLP] = ...
            adamupdate(MLP,gradientvalue, ...
            trailingAvgMLP, trailingAvgSqMLP, iteration, ...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);
        
%         M=MLP.Layers(28,1).Weights;
%         PseudoM=inv(M'*M)*M';
%         tmp_net = MLP.saveobj;
%         Layers=tmp_net.LayerGraph.Layers;
%         Layers(26, 1).Weights  =PseudoM;
%         Layers(16, 1).Bias  =[zeros(size_H,1);M(:)];
%         lgraph = layerGraph(Layers(1:25,1));
%         
%         skipfc1 = fullyConnectedLayer(size_H,'Name','skipfc_1','Weights',Layers(26, 1).Weights,'Bias',1e-40.*ones(size_H,1),'BiasLearnRateFactor',0);
%         ratio1=fullyConnectedLayer(size_H,'Name','skipratio_1','Weights',Layers(27, 1).Weights,'Bias',1e-40.*ones(size_H,1),'BiasLearnRateFactor',0);
%         lgraph = addLayers(lgraph,skipfc1);
%         lgraph = addLayers(lgraph,ratio1);
%         lgraph = connectLayers(lgraph,'input','skipfc_1');
%         lgraph = connectLayers(lgraph,'skipfc_1','skipratio_1');
%         lgraph = connectLayers(lgraph,'skipratio_1','add_1/in2');
%         skipfc2 = fullyConnectedLayer(size_Y,'Name','skipfc_2','Weights',Layers(28, 1).Weights,'Bias',1e-40.*ones(size_Y,1),'BiasLearnRateFactor',0);
%         ratio2=fullyConnectedLayer(size_Y,'Name','skipratio_2','Weights',Layers(29, 1).Weights,'Bias',1e-40.*ones(size_Y,1),'BiasLearnRateFactor',0);
%         lgraph = addLayers(lgraph,skipfc2);
%         lgraph = addLayers(lgraph,ratio2);
%         lgraph = connectLayers(lgraph,'softmax_1','skipfc_2');
%         lgraph = connectLayers(lgraph,'skipfc_2','skipratio_2');
%         lgraph = connectLayers(lgraph,'skipratio_2','add_2/in2');
%         MLP = dlnetwork(lgraph);

        
        % Update the scores plot

%         addpoints(h,iteration,...
%             double(gather(extractdata(lossvalue))));
%         
%         % Update the title with training progress information.
%         D = duration(0,0,toc(start),'Format','hh:mm:ss');
%         title(...
%             "Epoch: " + epoch + ", " + ...
%             "Iteration: " + iteration + ", " + ...
%             "Elapsed: " + string(D))
%         
%         drawnow
    end
    past_loss = loss;
    loss = lossvalue;
    if epoch == 1
        past_loss = loss;
        loss_0 = loss;
    end
    c = abs(loss-past_loss)/loss_0;
    if c < stop_criterion && epoch > 5
        break
    end
end


encoderlayers = [
    featureInputLayer( inputsize,'Name','encoderinput','Normalization','none')
    fullyConnectedLayer(inputsize*2,'Name','encoderfc_11')
    leakyReluLayer('Name','relu_11')
    fullyConnectedLayer(round(inputsize/2),'Name','encoderfc_1')
    leakyReluLayer('Name','relu_1')
    fullyConnectedLayer(round(inputsize/8),'Name','encoderfc_2')   
    leakyReluLayer('Name','relu_2')
    fullyConnectedLayer(12,'Name','encoderfc_3')   
    leakyReluLayer('Name','relu_3')
    fullyConnectedLayer(size_H,'Name','encoderfc_4')
    leakyReluLayer('Name','relu_4')
    fullyConnectedLayer(size_H,'Name','encoderratio_1','Weights',init_factor .*ones(size_H,size_H),'Bias',1e-40.*ones(size_H,1))
    additionLayer(2,'Name','encoderadd_1')
    fullyConnectedLayer(size_H,'Name','encoderfc_add_1')
    UntilityLayer('softmax_1')
%     softmaxLayer('Name','softmax_1')
    
    fullyConnectedLayer(size_H,'Name','encoderfc_5','Weights',eye(size_H),'Bias',1e-40.*ones(size_H,1))
    regressionLayer('Name','regressionOutput')];
for index=2:2:14
    encoderlayers(index).Weights=MLP.Layers(index).Weights;
    encoderlayers(index).Bias=MLP.Layers(index).Bias;
    encoderlayers(index).WeightLearnRateFactor = 0;
    encoderlayers(index).BiasLearnRateFactor = 0;
end
encoderlayers(16).WeightLearnRateFactor = 0;
encoderlayers(16).BiasLearnRateFactor = 0;
encoderlgraph = layerGraph(encoderlayers);
encoderskipfc1 = fullyConnectedLayer(size_H,'Name','encoderskipfc_1','Weights',MLP.Layers(26).Weights,'Bias',zeros(size_H,1),'WeightLearnRateFactor',0,'BiasLearnRateFactor',0);
encoderratio1=fullyConnectedLayer(size_H,'Name','encoderskipratio_1','Weights',MLP.Layers(27).Weights,'Bias',zeros(size_H,1),'WeightLearnRateFactor',0,'BiasLearnRateFactor',0);
encoderlgraph = addLayers(encoderlgraph,encoderskipfc1);
encoderlgraph = addLayers(encoderlgraph,encoderratio1);
encoderlgraph = connectLayers(encoderlgraph,'encoderinput','encoderskipfc_1');
encoderlgraph = connectLayers(encoderlgraph,'encoderskipfc_1','encoderskipratio_1');
encoderlgraph = connectLayers(encoderlgraph,'encoderskipratio_1','encoderadd_1/in2');
encoderoptions = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',1, ...
    'InitialLearnRate',1e-4, ...1e-4
    'GradientThreshold',6,...
    'SquaredGradientDecayFactor',0.99,...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',15, ...
    'Shuffle','every-epoch', ...'ValidationData',{input2,output2}, ...'ValidationFrequency',validationFrequency, ...'ValidationPatience',3, ...
    'Plots','none', ...'Plots','training-progress',...
    'Verbose',false);
encoderMLP = trainNetwork(input', H',encoderlgraph,encoderoptions);
% Most probable regions
outputest=predict(encoderMLP,input');
Hest=outputest';

%% =======================================================================
layers_final= [
    featureInputLayer(inputsize,'Name','input','Normalization','none')  %featureinputlayer  comment normalization
    fullyConnectedLayer(inputsize*2,'Name','fc_11')   % 2 layers   initialization with small value instead of 0.
    leakyReluLayer('Name','relu_11')
    fullyConnectedLayer(round(inputsize/2),'Name','fc_1')   
    leakyReluLayer('Name','relu_1')
    fullyConnectedLayer(round(inputsize/8),'Name','fc_2')   
    leakyReluLayer('Name','relu_2')
    fullyConnectedLayer(12,'Name','fc_3')  
    leakyReluLayer('Name','relu_3')
    fullyConnectedLayer(size_H,'Name','fc_4')
    leakyReluLayer('Name','relu_4')
    fullyConnectedLayer(size_H,'Name','ratio_1','Weights',init_factor *eye(size_H),'Bias',1e-40.*ones(size_H,1),'BiasLearnRateFactor',0)
    additionLayer(2,'Name','add_1')
    fullyConnectedLayer(size_H,'Name','fc_add_1','Weights',eye(size_H),'Bias',1e-40.*ones(size_H,1),'WeightLearnRateFactor',0,'BiasLearnRateFactor',0)
%     softmaxLayer('Name','softmax_1')
    UntilityLayer('softmax_1')
    
    fullyConnectedLayer(size_M,'Name','concatenate','Weights',Weight_concat,'Bias',Bias_concat,'WeightLearnRateFactor',0,'BiasLearnRateFactor',0)
    fullyConnectedLayer(size_H*size_Y,'Name','fc_5')
    leakyReluLayer('Name','relu_5')
    fullyConnectedLayer(size_Y,'Name','fc_6')
    leakyReluLayer('Name','relu_6')
    fullyConnectedLayer(size_Y,'Name','fc_7')
    leakyReluLayer('Name','relu_7')
    fullyConnectedLayer(size_Y,'Name','ratio_2','Weights',init_factor.*ones(size_Y,size_Y),'Bias',1e-40.*ones(size_Y,1),'BiasLearnRateFactor',0)
    additionLayer(2,'Name','add_2')
    fullyConnectedLayer(size_Y,'Name','fc_add_2','Weights',eye(size_Y),'Bias',1e-40.*ones(size_Y,1),'WeightLearnRateFactor',0,'BiasLearnRateFactor',0)
    reluLayer('Name', 'relu_8')
    regressionLayer('Name','regressionOutput')
    ];
for ind=[2:2:16,17:2:25]
    layers_final(index).Weights=MLP.Layers(index).Weights;
    layers_final(index).Bias=MLP.Layers(index).Bias;
    layers_final(index).WeightLearnRateFactor = 0;
    layers_final(index).BiasLearnRateFactor = 0;
end
lgraph_final = layerGraph(layers_final);
skipfc1_final = fullyConnectedLayer(size_H,'Name','skipfc_1','Weights',MLP.Layers(26).Weights,'Bias',1e-40.*ones(size_H,1),'WeightLearnRateFactor',0,'BiasLearnRateFactor',0);
ratio1_final=fullyConnectedLayer(size_H,'Name','skipratio_1','Weights',MLP.Layers(27).Weights,'Bias',1e-40.*ones(size_H,1),'WeightLearnRateFactor',0,'BiasLearnRateFactor',0);
lgraph_final = addLayers(lgraph_final,skipfc1_final);
lgraph_final = addLayers(lgraph_final,ratio1_final);
lgraph_final = connectLayers(lgraph_final,'input','skipfc_1');
lgraph_final = connectLayers(lgraph_final,'skipfc_1','skipratio_1');
lgraph_final = connectLayers(lgraph_final,'skipratio_1','add_1/in2');

skipfc2_final = fullyConnectedLayer(size_Y,'Name','skipfc_2','Weights',MLP.Layers(28).Weights,'Bias',1e-40.*ones(size_Y,1),'WeightLearnRateFactor',0,'BiasLearnRateFactor',0);
ratio2_final=fullyConnectedLayer(size_Y,'Name','skipratio_2','Weights',MLP.Layers(29).Weights,'Bias',1e-40.*ones(size_Y,1),'WeightLearnRateFactor',0,'BiasLearnRateFactor',0);
lgraph_final = addLayers(lgraph_final,skipfc2_final);
lgraph_final = addLayers(lgraph_final,ratio2_final);
lgraph_final = connectLayers(lgraph_final,'softmax_1','skipfc_2');
lgraph_final = connectLayers(lgraph_final,'skipfc_2','skipratio_2');
lgraph_final = connectLayers(lgraph_final,'skipratio_2','add_2/in2');
finalMLP = trainNetwork(input', input',lgraph_final,encoderoptions);
Yest=predict(finalMLP,input');
rmse=sqrt(norm(Y-Yest','fro')^2);
end

function [gradientvalue, lossvalue] = modelGradients(MLP, dlX,lambda1,lambda2,lambda3,lambda5,lambda6,M0)

% Calculate the predictions for real data with the discriminator network.
dlYPred = forward(MLP, dlX);

% Convert the discriminator outputs to probabilities.
Yest = relu(dlYPred);
weights1=MLP.Learnables.Value{11,1};
weight1skip=MLP.Learnables.Value{15,1};
weights2=MLP.Learnables.Value{27,1};
weights2skip=MLP.Learnables.Value{31,1};
offdiag1=sum(sum(abs(weights1-eye(length(weights1)).*weights1)));
offdiag1skip=sum(sum(abs(weight1skip-eye(length(weight1skip)).*weight1skip)));
offdiag2=sum(sum(abs(weights2-eye(length(weights2)).*weights2)));
offdiag2skip=sum(sum(abs(weights2skip-eye(length(weights2skip)).*weights2skip)));
weightsum=0;
for i=[1:2:11,21:2:27]
    weightsum=weightsum+sum(sum(abs(MLP.Learnables.Value{i,1})));
end
biassum=0;
for i=[2:2:12,22:2:28]
    biassum=biassum+sum(sum(abs(MLP.Learnables.Value{i,1})));
end
M=MLP.Learnables.Value{29,1};
SAD=0;
for i=1:size(M,2)
    SAD=SAD+(abs(M(:,i)'*M0(:,i)/(sqrt(sum(M(:,i).^2))*sqrt(sum(M0(:,i).^2)))));
end
M=MLP.Learnables.Value{29,1};
pseudoM=MLP.Learnables.Value{13,1};
MSE=sum(sum(abs((M'*M)*pseudoM-M')));
% lossvalue=mse(Yest,dlX)+lambda1*(offdiag1skip+offdiag1+offdiag2)+lambda2*offdiag2skip+lambda3*weightsum+lambda4*biassum+lambda5*SAD; 
lossvalue=mse(Yest,dlX)+lambda1*(offdiag1skip+offdiag1+offdiag2)+lambda2*offdiag2skip+lambda3*(weightsum+biassum)+lambda5*MSE+lambda6*SAD; 
% For each network, calculate the gradients with respect to the loss.  
gradientvalue = dlgradient(lossvalue, MLP.Learnables);

end