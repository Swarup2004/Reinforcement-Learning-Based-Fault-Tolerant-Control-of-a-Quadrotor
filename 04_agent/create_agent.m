function agent = create_agent(env)
% create_agent.m
% Creates DDPG agent for quadrotor fault-tolerant control
%
% DDPG = Deep Deterministic Policy Gradient
%
% ACTOR:  maps observation (9x1) -> action (4x1) via tanh
%   Architecture: 9 -> 128 -> 128 -> 4 (tanh)
%   Output in [-1,1] — decoded to motor omega^2 in env.step()
%
% CRITIC: maps (observation, action) -> Q scalar
%   Architecture: obs(9)->64, act(4)->64, merged->128->1
%
% DDPG UPDATE EQUATIONS:
%   Target:  y = r + gamma * Q_target(s_next, pi_target(s_next))
%   Critic:  min  E[(Q(s,a) - y)^2]
%   Actor:   max  E[Q(s, pi(s))]
%   Soft update: theta_target <- tau*theta + (1-tau)*theta_target

    obsInfo = getObservationInfo(env);
    actInfo = getActionInfo(env);
    nObs    = prod(obsInfo.Dimension);  % 9
    nAct    = prod(actInfo.Dimension);  % 4

    %--- ACTOR NETWORK ---
    actorNet = [
        featureInputLayer(nObs,'Normalization','none','Name','obs')
        fullyConnectedLayer(128,'Name','fc1')
        reluLayer('Name','relu1')
        fullyConnectedLayer(128,'Name','fc2')
        reluLayer('Name','relu2')
        fullyConnectedLayer(nAct,'Name','fc_out')
        tanhLayer('Name','tanh_out')
    ];
    actorNet = dlnetwork(actorNet);
    actor = rlContinuousDeterministicActor(actorNet, obsInfo, actInfo, ...
        'ObservationInputNames', 'obs');

    %--- CRITIC NETWORK ---
    % Two input paths merged via addition
    statePath = [
        featureInputLayer(nObs,'Normalization','none','Name','obs')
        fullyConnectedLayer(64,'Name','sfc1')
        reluLayer('Name','srelu1')
        fullyConnectedLayer(64,'Name','sfc2')
    ];
    actionPath = [
        featureInputLayer(nAct,'Normalization','none','Name','act')
        fullyConnectedLayer(64,'Name','afc1')
    ];
    commonPath = [
        additionLayer(2,'Name','add')
        reluLayer('Name','crelu1')
        fullyConnectedLayer(128,'Name','cfc1')
        reluLayer('Name','crelu2')
        fullyConnectedLayer(1,'Name','Q_out')
    ];
    criticGraph = layerGraph();
    criticGraph = addLayers(criticGraph, statePath);
    criticGraph = addLayers(criticGraph, actionPath);
    criticGraph = addLayers(criticGraph, commonPath);
    criticGraph = connectLayers(criticGraph, 'sfc2',  'add/in1');
    criticGraph = connectLayers(criticGraph, 'afc1',  'add/in2');
    criticNet = dlnetwork(criticGraph);
    critic = rlQValueFunction(criticNet, obsInfo, actInfo, ...
        'ObservationInputNames', 'obs', ...
        'ActionInputNames',      'act');

    %--- DDPG OPTIONS ---
    agentOpts = rlDDPGAgentOptions;
    agentOpts.ExperienceBufferLength          = 100000;
    agentOpts.MiniBatchSize                   = 128;
    agentOpts.DiscountFactor                  = 0.99;
    agentOpts.TargetSmoothFactor              = 0.005;
    agentOpts.TargetUpdateFrequency           = 1;
    agentOpts.CriticOptimizerOptions.LearnRate = 1e-3;
    agentOpts.ActorOptimizerOptions.LearnRate  = 1e-4;
    agentOpts.NoiseOptions.Variance           = 0.3;
    agentOpts.NoiseOptions.VarianceDecayRate  = 1e-5;
    agentOpts.ResetExperienceBufferBeforeTraining = true;

    %--- CREATE AGENT ---
    agent = rlDDPGAgent(actor, critic, agentOpts);

    fprintf('DDPG Agent created\n')
    fprintf('  Actor:  9->128->128->4 (tanh)\n')
    fprintf('  Critic: (9->64)+(4->64)->128->1\n')
    fprintf('  Buffer: %d  Batch: %d\n', ...
        agentOpts.ExperienceBufferLength, agentOpts.MiniBatchSize)
end
