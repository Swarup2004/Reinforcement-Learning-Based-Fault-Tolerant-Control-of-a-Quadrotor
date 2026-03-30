function agent = create_large_agent(env)
    obsInfo = getObservationInfo(env);
    actInfo = getActionInfo(env);
    nObs = prod(obsInfo.Dimension);
    nAct = prod(actInfo.Dimension);
    actorNet = [
        featureInputLayer(nObs,'Normalization','none','Name','obs')
        fullyConnectedLayer(256,'Name','fc1')
        reluLayer('Name','r1')
        fullyConnectedLayer(256,'Name','fc2')
        reluLayer('Name','r2')
        fullyConnectedLayer(128,'Name','fc3')
        reluLayer('Name','r3')
        fullyConnectedLayer(nAct,'Name','fc_out')
        tanhLayer('Name','tanh_out')
    ];
    actorNet = dlnetwork(actorNet);
    actor = rlContinuousDeterministicActor(actorNet,obsInfo,actInfo,...
        'ObservationInputNames','obs');
    statePath = [
        featureInputLayer(nObs,'Normalization','none','Name','obs')
        fullyConnectedLayer(256,'Name','sfc1')
        reluLayer('Name','sr1')
        fullyConnectedLayer(128,'Name','sfc2')
        reluLayer('Name','sr2')
    ];
    actionPath = [
        featureInputLayer(nAct,'Normalization','none','Name','act')
        fullyConnectedLayer(128,'Name','afc1')
        reluLayer('Name','ar1')
    ];
    commonPath = [
        additionLayer(2,'Name','add')
        reluLayer('Name','cr1')
        fullyConnectedLayer(256,'Name','cfc1')
        reluLayer('Name','cr2')
        fullyConnectedLayer(128,'Name','cfc2')
        reluLayer('Name','cr3')
        fullyConnectedLayer(1,'Name','Q_out')
    ];
    cg = layerGraph();
    cg = addLayers(cg,statePath);
    cg = addLayers(cg,actionPath);
    cg = addLayers(cg,commonPath);
    cg = connectLayers(cg,'sr2','add/in1');
    cg = connectLayers(cg,'ar1','add/in2');
    criticNet = dlnetwork(cg);
    critic = rlQValueFunction(criticNet,obsInfo,actInfo,...
        'ObservationInputNames','obs',...
        'ActionInputNames','act');
    opts = rlDDPGAgentOptions;
    opts.ExperienceBufferLength           = 200000;
    opts.MiniBatchSize                    = 256;
    opts.DiscountFactor                   = 0.99;
    opts.TargetSmoothFactor               = 0.005;
    opts.TargetUpdateFrequency            = 1;
    opts.CriticOptimizerOptions.LearnRate = 3e-4;
    opts.ActorOptimizerOptions.LearnRate  = 1e-4;
    opts.NoiseOptions.Variance            = 0.3;
    opts.NoiseOptions.VarianceDecayRate   = 1e-5;
    opts.ResetExperienceBufferBeforeTraining = true;
    agent = rlDDPGAgent(actor,critic,opts);
    fprintf('Large DDPG Agent: 9->256->256->128->4\n')
end
