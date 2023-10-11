%% DETC-K
clear all; close all; 

rng(1)

A = 4;
T = 100000 ;                   % total number
M = 1 ;                    % batch size
N = 4;                     % offline number
mu = [0.25, 0.25,0.25,0.25];            % data coverage
w = 1;                    % weight
alpha = 1;

r = [0.7, 0.2, 0.2, 0.2];                       % reward
r_opt = max(r);
Deltamin = max(r) - max(r(r<max(r)));
iternum = 1000;
logbns = log(T) + 0.2*sqrt(log(T));
Monitor = 10000;


R_off = zeros(A,iternum,N);                 % offline data
R_off(1,:,:) = randn(iternum,N) + r(1);
R_off(2,:,:) = randn(iternum,N) + r(2);
R_off(3,:,:) = randn(iternum,N) + r(3);
R_off(4,:,:) = randn(iternum,N) + r(4);

%%
Regret_record_DETC = zeros(iternum,T);
for iter = 1:iternum
    rng(iter)
    
    r_bar_DETC = mean(R_off(:,iter,:),3);
    Nk_DETC = N * mu';                          % number of pulls
    Regret_DETC = 0;
    
    s = 0;
    Stage = 1;
    Fail_Flag = 0;
    for t = 1:T    
        if Stage == 1
            R = randn(4,1) + r';
    
            action = mod(t-1, A) + 1;
    
            Reward_DETC = R(action);
            Regret_DETC = Regret_DETC + (r_opt - r(action));
            Regret_record_DETC(iter,t) = Regret_DETC;
    
            r_bar_DETC(action) = (r_bar_DETC(action) * Nk_DETC(action) + Reward_DETC) / (Nk_DETC(action) + 1);
            Nk_DETC(action) = Nk_DETC(action) + 1;
    
            if t > A*sqrt(log(T))
                Stage = 2;
                [~,action_cur] = max(r_bar_DETC);
                s = 0;
                r_p_DETC = 0;
            end
    
        elseif Stage == 2
            R = randn(4,1) + r';
            action = action_cur;
    
            Reward_DETC = R(action);
            Regret_DETC = Regret_DETC + (r_opt - r(action));
            Regret_record_DETC(iter,t) = Regret_DETC;
    
            r_p_DETC = (s * r_p_DETC + Reward_DETC) / (s+1);
            s = s+1;
            r_bar_DETC(action) = (r_bar_DETC(action) * Nk_DETC(action) + Reward_DETC) / (Nk_DETC(action) + 1);
            Nk_DETC(action) = Nk_DETC(action) + 1;
    
            if s > log(T)^2
                Stage = 3;
                ActionSet = [1:A];
                SubActionSet = ActionSet(~ismember([1:A],action_cur));
                break;
            end
        end
    end
    theta_bar = zeros(1,A-1);
    ti = zeros(1,A-1);


    for subidx = 1:A-1
        while t <= T
            R = randn(4,1) + r';
            action = SubActionSet(subidx);
        
            Reward_DETC = R(action);
            Regret_DETC = Regret_DETC + (r_opt - r(action));
            Regret_record_DETC(iter,t) = Regret_DETC;
            theta_bar(subidx) = (ti(subidx) * theta_bar(subidx) + Reward_DETC) / (ti(subidx) + 1);
            t = t+1;
            ti(subidx) = ti(subidx) + 1;
        
            if ti(subidx) > log(T)^2 || abs(r_p_DETC - theta_bar(subidx)) >= sqrt(2 / ti(subidx) * log(T/ti(subidx) * (log(T / ti(subidx))^2 + 1) ) )
                break;
            end
        end
        if t > T 
            break;
        end
        if ti(subidx) > log(T)^2
            Fail_Flag = 1;
            break;
        else
            continue;
        end
    end
    
    if t < T
        [~, Bestidx] = max(theta_bar);
        BestSubAction = SubActionSet(Bestidx);
        if r_bar_DETC(action_cur) >= theta_bar(Bestidx) && Fail_Flag == 0
            BestAction = action_cur;
            while t <= T
                R = randn(4,1) + r';
                action = BestAction;
    
                Reward_DETC = R(action);
                Regret_DETC = Regret_DETC + (r_opt - r(action));
                Regret_record_DETC(iter,t) = Regret_DETC;
                t = t + 1 ;
            end
        else
            r_bar_repull = zeros(1,4);
            Nk_repull = zeros(1,4);
            for idx = 1:A
                while t <= T
                    R = randn(4,1) + r';
                    action = idx;
                    Reward_DETC = R(action);
                    Regret_DETC = Regret_DETC + (r_opt - r(action));
                    Regret_record_DETC(iter,t) = Regret_DETC;
                    t = t + 1 ;

                    r_bar_repull(idx) = (r_bar_repull(idx) * Nk_repull(idx) + Reward_DETC) / (Nk_repull(idx) + 1);
                    Nk_repull(idx) = Nk_repull(idx) + 1;
    
                    if Nk_repull(idx) > log(T)^2
                        break;
                    end
                end
                if t> T
                    break;
                end
            end
            
            [~,BestAction] = max(r_bar_repull);
            while t<= T
                R = randn(4,1) + r';
                action = BestAction;
                Reward_DETC = R(action);
                Regret_DETC = Regret_DETC + (r_opt - r(action));
                Regret_record_DETC(iter,t) = Regret_DETC;
                t = t + 1 ;
            end
        end
    end
end
Regret_mean_DETC = mean(Regret_record_DETC, 1);
%%
save('DETCK.mat','Regret_mean_DETC');