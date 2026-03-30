classdef BalancedFaultyEnv < QuadrotorEnv
% Cycles through all 4 motors systematically
% Ensures equal training on every motor fault
    properties
        EpisodeCounter = 0
    end
    methods
        function this = BalancedFaultyEnv()
            this = this@QuadrotorEnv();
            this.w_z      = 5.0;
            this.w_vz     = 1.0;
            this.w_angle  = 0.5;
            this.w_action = 0.01;
        end
        function InitObs = reset(this)
            InitObs = reset@QuadrotorEnv(this);
            this.EpisodeCounter = this.EpisodeCounter + 1;
            % Cycle: ep1=M1, ep2=M2, ep3=M3, ep4=M4, ep5=healthy
            % Every 5 episodes covers all motors + one healthy
            cycle = mod(this.EpisodeCounter, 5);
            this.Lambda = ones(4,1);
            if cycle ~= 0
                % cycle 1-4 = fault on motor 1-4
                severity = 0.2 + rand*0.6; % LoE in [0.2, 0.8]
                this.Lambda(cycle) = severity;
            end
            % Log what fault was injected
        end
    end
end
