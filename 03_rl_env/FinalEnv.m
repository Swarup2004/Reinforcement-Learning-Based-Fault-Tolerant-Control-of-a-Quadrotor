classdef FinalEnv < QuadrotorEnv
    properties
        EpCount = 0
    end
    methods
        function this = FinalEnv()
            this = this@QuadrotorEnv();
            this.w_z      = 5.0;
            this.w_vz     = 1.0;
            this.w_angle  = 0.3;
            this.w_action = 0.005;
            this.MaxTilt  = pi/2;
        end
        function InitObs = reset(this)
            InitObs = reset@QuadrotorEnv(this);
            this.EpCount = this.EpCount + 1;
            % Strict cycle: M1,M2,M3,M4,healthy repeated
            cycle = mod(this.EpCount,5);
            this.Lambda = ones(4,1);
            if cycle ~= 0
                sev = 0.3 + rand*0.5; % LoE in [0.3,0.8]
                this.Lambda(cycle) = sev;
            end
        end
    end
end
