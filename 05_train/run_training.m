% run_training.m
% Main training script
% Run this file to train the DDPG agent
% Expected time: 30-90 min depending on CPU

clear; clc;
cd('C:\Users\ONGC\RL_for_quadrotors')

% Add all folders to path
addpath('01_params')
addpath('03_rl_env')
addpath('04_agent')
addpath('06_results')

% Create environment
fprintf('Creating environment...\n')
env = QuadrotorEnv();
validateEnvironment(env);
fprintf('Environment OK\n')

% Create agent
fprintf('Creating DDPG agent...\n')
agent = create_agent(env);

% Training options
trainOpts = rlTrainingOptions;
trainOpts.MaxEpisodes                  = 2000;
trainOpts.MaxStepsPerEpisode           = 1000;
trainOpts.ScoreAveragingWindowLength   = 50;
trainOpts.StopTrainingCriteria         = 'AverageReward';
trainOpts.StopTrainingValue            = 150;
trainOpts.Verbose                      = true;
trainOpts.Plots                        = 'training-progress';

% Save best agent automatically
if ~exist('05_train\saved_agents','dir')
    mkdir('05_train\saved_agents')
end
trainOpts.SaveAgentCriteria            = 'EpisodeReward';
trainOpts.SaveAgentValue               = 50;
trainOpts.SaveAgentDirectory           = '05_train\saved_agents';

% Use parallel training (you have Parallel Computing Toolbox)
trainOpts.UseParallel                  = true;

% Train
fprintf('\nStarting training...\n')
fprintf('MaxEpisodes: %d  MaxSteps: %d\n', ...
    trainOpts.MaxEpisodes, trainOpts.MaxStepsPerEpisode)
fprintf('Stop when 50-episode avg reward > %d\n', ...
    trainOpts.StopTrainingValue)

trainingStats = train(agent, env, trainOpts);

% Save final agent and stats
save('05_train\trained_agent.mat', 'agent', 'trainingStats')
fprintf('\nTraining complete. Agent saved.\n')
fprintf('Best episode reward: %.2f\n', max(trainingStats.EpisodeReward))
fprintf('Run 06_results\plot_results.m to evaluate.\n')
