classdef DiagFaultyEnv < QuadrotorEnv
% Targets Motor2 and Motor4 specifically
% These are the motors Stage 2 agent failed on
    properties
        EpisodeCounter = 0
    end
    methods
        function this = DiagFaultyEnv()
            this = this@QuadrotorEnv();
            this.w_z      = 5.0;
            this.w_vz     = 1.0;
            this.w_angle  = 0.5;
            this.w_action = 0.01;
        end
        function InitObs = reset(this)
            InitObs = reset@QuadrotorEnv(this);
            this.EpisodeCounter = this.EpisodeCounter + 1;
            % Pattern: M2, M4, M2, M4, healthy, M1, M3, healthy
            % Heavy focus on M2 and M4
            cycle = mod(this.EpisodeCounter, 8);
            this.Lambda = ones(4,1);
            severity = 0.2 + rand*0.6;
            if     cycle == 1; this.Lambda(2) = severity;
            elseif cycle == 2; this.Lambda(4) = severity;
            elseif cycle == 3; this.Lambda(2) = severity;
            elseif cycle == 4; this.Lambda(4) = severity;
            elseif cycle == 5; this.Lambda = ones(4,1); % healthy
            elseif cycle == 6; this.Lambda(1) = severity;
            elseif cycle == 7; this.Lambda(3) = severity;
            elseif cycle == 0; this.Lambda = ones(4,1); % healthy
            end
        end
    end
end
