classdef M2M4Env < QuadrotorEnv
    properties
        EpCount = 0
    end
    methods
        function this = M2M4Env()
            this = this@QuadrotorEnv();
            this.w_z      = 5.0;
            this.w_vz     = 1.0;
            this.w_angle  = 0.5;
            this.w_action = 0.01;
            this.MaxTilt  = pi/2;
        end
        function InitObs = reset(this)
            InitObs = reset@QuadrotorEnv(this);
            this.EpCount = this.EpCount + 1;
            % Strict alternation: M2, M4, M2, M4, healthy
            cycle = mod(this.EpCount, 5);
            this.Lambda = ones(4,1);
            sev = 0.3 + rand*0.5;
            if     cycle==1; this.Lambda(2)=sev;
            elseif cycle==2; this.Lambda(4)=sev;
            elseif cycle==3; this.Lambda(2)=sev;
            elseif cycle==4; this.Lambda(4)=sev;
            else;            this.Lambda=ones(4,1);
            end
        end
    end
end
