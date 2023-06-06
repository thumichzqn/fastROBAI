clear all; close all; 

rng(1)

A = 2;
T = 100000;                   % total
M = 1 ;                    % batch size
N = 2;                     % offline number
mu = [0.5, 0.5];            % data coverage
w = 1;                    % weight
alpha = 1;

r = [0.7, 0.2];                       % reward mean
r_opt = max(r);                     
iternum = 100000;
logbns = log(T) + sqrt(log(T));


R_off = zeros(A,iternum,N);                 % offline data
R_off(1,:,:) = double(rand(iternum,N) < r(1));
R_off(2,:,:) = double(rand(iternum,N) < r(2));

%% Prop Parameter
diary 'mylog4.txt'
rng(1)

Nk = N * mu' .* ones(2,iternum);                          % number of pulls
    
r_bar = mean(R_off,3);   % empirical mean
bk = sqrt(2 * alpha * logbns ./ Nk);                             % bonus
Decision_k = r_bar + bk;
Regret_record = zeros(1,T);
Regret = zeros(1,iternum);

for t=1:T
    
    R = zeros(2, iternum, 1);                   % online sample
    R(1,:,:) = double(rand(iternum,1) < r(1));
    R(2,:,:) = double(rand(iternum,1) < r(2));

    %% Prop Simulation

    if t < 16/(r(1)-r(2))^2 * alpha  * logbns
        Decision_k = r_bar + bk;                            % bonus
                                 
    elseif t-1 < 16/(r(1)-r(2))^2 * alpha  * logbns
        Decision_k = r_bar - bk;
    end

    [~,a_choose] = max(Decision_k); 
    a_choose_mask = double((Decision_k) > flip(Decision_k,1));
    reward = R(:,:,1) .* a_choose_mask;

    Regret = Regret + (r_opt - r(a_choose));

    r_bar = (r_bar .* Nk + reward) ./ (Nk + a_choose_mask);          % empirical mean
    Nk = Nk + a_choose_mask;
    bk = sqrt(2 * alpha * logbns ./ Nk);
    Regret_record(t) = mean(Regret);
    if mod(t, 10000) == 0     
        diary on
        disp(['t = ', num2str(t), ', Regret: ', num2str(mean(Regret))])
        diary off
    end
    
end
Final_Sub_Pull = Nk(2,:)';
Final_Reg = Regret;

diary on
disp(['EOCP:   Regret:', num2str(mean(Final_Reg)), ',STD:', num2str(std(Final_Sub_Pull))])
diary off


%% ug Parameter
rng(1)

Nk_UG = N * mu' .* ones(2,iternum);                          % number of pulls
    
r_bar_UG = mean(R_off,3);   % empirical mean
bk_UG = sqrt(2 * alpha * logbns ./ Nk_UG);                             % bonus
Decision_k_UG = r_bar_UG + bk_UG;
Regret_record_UG = zeros(1,T);
Regret_UG = zeros(1,iternum);
Stop_UG = zeros(2,iternum);

for t=1:T
    
    R = zeros(2, iternum, 1);                   % online sample
    R(1,:,:) = double(rand(iternum,1) < r(1));
    R(2,:,:) = double(rand(iternum,1) < r(2));

    %% Prop Simulation

    Stopping_UG = repmat(double((Nk_UG(1,:) ./ Nk_UG(2,:)) > logbns) + double((Nk_UG(2,:) ./ Nk_UG(1,:)) > logbns), 2,1) .* (1 - Stop_UG);
    Stop_UG = Stop_UG + Stopping_UG;
    Decision_k_UG = Stop_UG .* Decision_k_UG + (1 - Stop_UG).* (r_bar_UG + bk_UG);
    Decision_k_UG = Stopping_UG .* (r_bar_UG - bk_UG) + (1 - Stopping_UG).* Decision_k_UG;

    [~,a_choose_UG] = max(Decision_k_UG); 
    a_choose_mask_UG = double((Decision_k_UG) > flip(Decision_k_UG,1));
    reward_UG = R(:,:,1) .* a_choose_mask_UG;

    Regret_UG = Regret_UG + (r_opt - r(a_choose_UG));

    r_bar_UG = (r_bar_UG .* Nk_UG + reward_UG) ./ (Nk_UG + a_choose_mask_UG);          % empirical mean
    Nk_UG = Nk_UG + a_choose_mask_UG;
    bk_UG = sqrt(2 * alpha * logbns ./ Nk_UG); 
    Regret_record_UG(t) = mean(Regret_UG);
    if mod(t, 10000) == 0     
        diary on
        disp(['t = ', num2str(t), ', Regret: ', num2str(mean(Regret_UG))])
        diary off
    end
    
end

Final_Sub_Pull_UG = Nk_UG(2,:)';
Final_Reg_UG = Regret_UG;

diary on
disp(['UG:   Regret:', num2str(mean(Final_Reg_UG)), ',STD:', num2str(std(Final_Sub_Pull_UG))])
diary off

%% kl-eocp-fixed
rng(1)

KL_mean1 = r(2) .* log(r(2)/r(1)) + (1-r(2)).* log( (1-r(2))./(1-r(1)) ) ;
[~,ra_prime] = invKL_ULCB(r(1),1,KL_mean1/4); 
KL_mean2 = 4 * (ra_prime .* log(ra_prime/r(2) ) + (1-ra_prime) .* log( (1-ra_prime)./(1-r(2)) ) ) ;
KL_mean = min(KL_mean1, KL_mean2);

Nk_kl = N * mu' .* ones(2,iternum);                          % number of pulls
    
r_bar_kl = mean(R_off,3);   % empirical mean
[UCB_kl, LCB_kl] = KL_ULCB(r_bar_kl, Nk_kl, logbns);

Decision_k_kl = UCB_kl;
Regret_record_kl = zeros(1,T);
Regret_kl = zeros(1,iternum);

for t=1:T
    
    R = zeros(2, iternum, 1);                   % online sample
    R(1,:,:) = double(rand(iternum,1) < r(1));
    R(2,:,:) = double(rand(iternum,1) < r(2));


    if t < 8/KL_mean * alpha  * logbns
        Decision_k_kl = UCB_kl;                            % bonus
                                 
    elseif t-1 < 8/KL_mean * alpha  * logbns
        Decision_k_kl = LCB_kl;
    end

    [~,a_choose_kl] = max(Decision_k_kl); 
    a_choose_mask_kl = double((Decision_k_kl) > flip(Decision_k_kl,1));
    reward_kl = R(:,:,1) .* a_choose_mask_kl;

    Regret_kl = Regret_kl + (r_opt - r(a_choose_kl));

    r_bar_kl = (r_bar_kl .* Nk_kl + reward_kl) ./ (Nk_kl + a_choose_mask_kl);          % empirical mean
    Nk_kl = Nk_kl + a_choose_mask_kl;
    [UCB_kl, LCB_kl] = KL_ULCB(r_bar_kl, Nk_kl, logbns);
    Regret_record_kl(t) = mean(Regret_kl);
    if mod(t, 10000) == 0     
        diary on
        disp(['t = ', num2str(t), ', Regret: ', num2str(mean(Regret_kl))])
        diary off
    end
    
end

Final_Sub_Pull_kl = Nk_kl(2,:)';
Final_Reg_kl = Regret_kl;

diary on
disp(['KL-EOCP:   Regret:', num2str(mean(Final_Reg_kl)), ',STD:', num2str(std(Final_Sub_Pull_kl))])
diary off

%% KL-EOCP-UG
rng(1)

Nk_klug = N * mu' .* ones(2,iternum);                          % number of pulls
    
r_bar_klug = mean(R_off,3);   % empirical mean
[UCB_klug, LCB_klug] = KL_ULCB(r_bar_klug, Nk_klug, logbns);

Decision_k_klug = UCB_klug;
Regret_record_klug = zeros(1,T);
Regret_klug = zeros(1,iternum);
Stop_klug = zeros(2,iternum);

for t=1:T
    
    R = zeros(2, iternum, 1);                   % online sample
    R(1,:,:) = double(rand(iternum,1) < r(1));
    R(2,:,:) = double(rand(iternum,1) < r(2));

    Stopping_klug = repmat(double((Nk_klug(1,:) ./ Nk_klug(2,:)) > logbns) ...
        + double((Nk_klug(2,:) ./ Nk_klug(1,:)) > logbns), 2,1) .* (1 - Stop_klug);
    Stop_klug = Stop_klug + Stopping_klug;
    Decision_k_klug = Stop_klug .* Decision_k_klug + (1 - Stop_klug).* UCB_klug;
    Decision_k_klug = Stopping_klug .* LCB_klug + (1 - Stopping_klug).* Decision_k_klug;

    [~,a_choose_klug] = max(Decision_k_klug); 
    a_choose_mask_klug = double((Decision_k_klug) > flip(Decision_k_klug,1));
    reward_klug = R(:,:,1) .* a_choose_mask_klug;

    Regret_klug = Regret_klug + (r_opt - r(a_choose_klug));

    r_bar_klug = (r_bar_klug .* Nk_klug + reward_klug) ./ (Nk_klug + a_choose_mask_klug);          % empirical mean
    Nk_klug = Nk_klug + a_choose_mask_klug;
    [UCB_klug, LCB_klug] = KL_ULCB(r_bar_klug, Nk_klug, logbns);
    Regret_record_klug(t) = mean(Regret_klug);
    if mod(t, 10000) == 0     
        diary on
        disp(['t = ', num2str(t), ', Regret: ', num2str(mean(Regret_klug))])
        diary off
    end
    
end

Final_Sub_Pull_klug = Nk_klug(2,:)';
Final_Reg_klug = Regret_klug;

diary on
disp(['KL-EOCP-UG:   Regret:', num2str(mean(Final_Reg_klug)), ',STD:', num2str(std(Final_Sub_Pull_klug))])
diary off

%%  UCB Parameters

if 1

rng(1)

Nk_UCB = N * mu' .* ones(2,iternum);                          % number of pulls
    
r_bar_UCB = mean(R_off,3);   % empirical mean
bk_UCB = sqrt(2 * alpha * log(T) ./ Nk_UCB);                             % bonus
Decision_k_UCB = r_bar_UCB + bk_UCB;
Regret_UCB = zeros(1,iternum);
Regret_record_UCB = zeros(1,T);

for t=1:T

    R = zeros(2, iternum, 1);                   % online sample
    R(1,:,:) = double(rand(iternum,1) < r(1));
    R(2,:,:) = double(rand(iternum,1) < r(2));

    Decision_k_UCB = r_bar_UCB + bk_UCB;                            % bonus
                                 
    [~,a_choose_UCB] = max(Decision_k_UCB); 
    a_choose_mask_UCB = double((Decision_k_UCB) > flip(Decision_k_UCB,1));
    reward_UCB = R(:,:,1) .* a_choose_mask_UCB;

    Regret_UCB = Regret_UCB + (r_opt - r(a_choose_UCB));

    r_bar_UCB = (r_bar_UCB .* Nk_UCB + reward_UCB) ./ (Nk_UCB + a_choose_mask_UCB);          % empirical mean
    Nk_UCB = Nk_UCB + a_choose_mask_UCB;
    bk_UCB = sqrt(2 * alpha * log(T) ./ Nk_UCB); 
    Regret_record_UCB(t) = mean(Regret_UCB);

    if mod(t, 10000) == 0     
        diary on
        disp(['t = ', num2str(t), ', Regret: ', num2str(mean(Regret_UCB))])
        diary off
    end

end



Final_Sub_Pull_UCB = Nk_UCB(2,:)';
Final_Reg_UCB = Regret_UCB;

diary on
disp(['UCB:   Regret:', num2str(mean(Final_Reg_UCB)), ',STD:', num2str(std(Final_Sub_Pull_UCB))])
diary off
end
%% KLUCB

if 1

rng(1)

Nk_klUCB = N * mu' .* ones(2,iternum);                          % number of pulls
r_bar_klUCB = mean(R_off,3);   % empirical mean
[UCB_klUCB, LCB_klUCB] = KL_ULCB(r_bar_klUCB, Nk_klUCB, log(T));

Decision_k_klUCB = UCB_klUCB;
Regret_klUCB = zeros(1,iternum);
Regret_record_klUCB = zeros(1,T);

for t=1:T

    R = zeros(2, iternum, 1);                   % online sample
    R(1,:,:) = double(rand(iternum,1) < r(1));
    R(2,:,:) = double(rand(iternum,1) < r(2));

    Decision_k_klUCB = UCB_klUCB;                            % bonus
                                 
    [~,a_choose_klUCB] = max(Decision_k_klUCB); 
    a_choose_mask_klUCB = double((Decision_k_klUCB) > flip(Decision_k_klUCB,1));
    reward_klUCB = R(:,:,1) .* a_choose_mask_klUCB;

    Regret_klUCB = Regret_klUCB + (r_opt - r(a_choose_klUCB));

    r_bar_klUCB = (r_bar_klUCB .* Nk_klUCB + reward_klUCB) ./ (Nk_klUCB + a_choose_mask_klUCB);          % empirical mean
    Nk_klUCB = Nk_klUCB + a_choose_mask_klUCB;
    [UCB_klUCB, LCB_klUCB] = KL_ULCB(r_bar_klUCB, Nk_klUCB, log(T));
    
    Regret_record_klUCB(t) = mean(Regret_klUCB);

    if mod(t, 10000) == 0     
        diary on
        disp(['t = ', num2str(t), ', Regret: ', num2str(mean(Regret_klUCB))])
        diary off
    end

end



Final_Sub_Pull_klUCB = Nk_klUCB(2,:)';
Final_Reg_klUCB = Regret_klUCB;

diary on
disp(['KLUCB:   Regret:', num2str(mean(Final_Reg_klUCB)), ',STD:', num2str(std(Final_Sub_Pull_klUCB))])
diary off
end



%% BAI

if 1

rng(1)

Nk_BAI = N * mu' .* ones(2,iternum);                          % number of pulls
    
r_bar_BAI = mean(R_off,3);   % empirical mean
bk_BAI = sqrt(2 * alpha * log(T) ./ Nk_BAI);                             % bonus
Decision_k_BAI = r_bar_BAI + bk_BAI;
Regret_record_BAI = zeros(1,T);
Regret_BAI = zeros(1,iternum);
Stop_BAI = zeros(2,iternum);

for t=1:T
    
    R = zeros(2, iternum, 1);                   % online sample
    R(1,:,:) = double(rand(iternum,1) < r(1));
    R(2,:,:) = double(rand(iternum,1) < r(2));


    Stopping_BAI = repmat( double( abs(r_bar_BAI(1,:) - r_bar_BAI(2,:)) > sqrt(8*log(T/t)/t) ), 2,1) .* (1 - Stop_BAI);
    Stop_BAI = Stop_BAI + Stopping_BAI;
    if mod(t,2) == 1
        Decision_k_BAI = Stop_BAI .* Decision_k_BAI + (1 - Stop_BAI).* [ones(1, iternum); zeros(1,iternum)];
        Decision_k_BAI = Stopping_BAI .* r_bar_BAI + (1 - Stopping_BAI).* Decision_k_BAI;
    else
        Decision_k_BAI = Stop_BAI .* Decision_k_BAI + (1 - Stop_BAI).* [zeros(1,iternum); ones(1, iternum)];
        Decision_k_BAI = Stopping_BAI .* r_bar_BAI + (1 - Stopping_BAI).* Decision_k_BAI;
    end

    [~,a_choose_BAI] = max(Decision_k_BAI); 
    a_choose_mask_BAI = double((Decision_k_BAI) > flip(Decision_k_BAI,1));
    reward_BAI = R(:,:,1) .* a_choose_mask_BAI;

    Regret_BAI = Regret_BAI + (r_opt - r(a_choose_BAI));

    r_bar_BAI = (r_bar_BAI .* Nk_BAI + reward_BAI) ./ (Nk_BAI + a_choose_mask_BAI);          % empirical mean
    Nk_BAI = Nk_BAI + a_choose_mask_BAI;
    bk_BAI = sqrt(2 * alpha * log(T) ./ Nk_BAI); 
    Regret_record_BAI(t) = mean(Regret_BAI);
    if mod(t, 10000) == 0     
        diary on
        disp(['t = ', num2str(t), ', Regret: ', num2str(mean(Regret_BAI))])
        diary off
    end
    
end

Final_Sub_Pull_BAI = Nk_BAI(2,:)';
Final_Reg_BAI = Regret_BAI;

diary on
disp(['BAI-ETC:   Regret:', num2str(mean(Final_Reg_BAI)), ',STD:', num2str(std(Final_Sub_Pull_BAI))])
diary off
end


%% DETC

if 1

rng(1)

Nk_DETC = N * mu' .* ones(2,iternum);                          % number of pulls
r_bar_DETC = mean(R_off,3);   % empirical mean
bk_DETC = sqrt(2 * alpha * log(T) ./ Nk_DETC);                             % bonus

t_DETC_2 = 1;
r_bar_DETC_2 = zeros(1,iternum);
r_prime_DETC = zeros(1,iternum);

Decision_k_DETC = r_bar_DETC + bk_DETC;
Regret_record_DETC = zeros(1,T);
Regret_DETC = zeros(1,iternum);
Stop_DETC = zeros(2,iternum);
Stop_DETC_2 = zeros(1,iternum);

for t=1:T
    
    R = zeros(2, iternum, 1);                   % online sample
    R(1,:,:) = double(rand(iternum,1) < r(1));
    R(2,:,:) = double(rand(iternum,1) < r(2));

    if t < ceil(log(T)^2) 
        Stopping_DETC = repmat( double( abs(r_bar_DETC(1,:) - r_bar_DETC(2,:)) > sqrt(16 * max(log((log(T))^2 / t),0) / t) ), 2,1) .* (1 - Stop_DETC);
        Stop_DETC = Stop_DETC + Stopping_DETC;
        if mod(t,2) == 1
            Decision_k_DETC = Stop_DETC .* Decision_k_DETC + (1 - Stop_DETC).* [ones(1, iternum); zeros(1,iternum)];
            Decision_k_DETC = Stopping_DETC .* r_bar_DETC + (1 - Stopping_DETC).* Decision_k_DETC;
        else
            Decision_k_DETC = Stop_DETC .* Decision_k_DETC + (1 - Stop_DETC).* [zeros(1,iternum); ones(1, iternum)];
            Decision_k_DETC = Stopping_DETC .* r_bar_DETC + (1 - Stopping_DETC).* Decision_k_DETC;
        end

        [~,a_choose_DETC] = max(Decision_k_DETC); 
        a_choose_mask_DETC = double((Decision_k_DETC) > flip(Decision_k_DETC,1));
        reward_DETC = R(:,:,1) .* a_choose_mask_DETC;

        Regret_DETC = Regret_DETC + (r_opt - r(a_choose_DETC));

        r_bar_DETC = (r_bar_DETC .* Nk_DETC + reward_DETC) ./ (Nk_DETC + a_choose_mask_DETC);          % empirical mean
        Nk_DETC = Nk_DETC + a_choose_mask_DETC;
        bk_DETC = sqrt(2 * alpha * log(T) ./ Nk_DETC); 
        Regret_record_DETC(t) = mean(Regret_DETC);

    elseif t == ceil(log(T)^2) 
        Decision_k_DETC = Stop_DETC .* Decision_k_DETC + (1 - Stop_DETC) .* r_bar_DETC;
        r_prime_DETC = max(r_bar_DETC);
        CurBest_k_DETC = double(flip(Decision_k_DETC,1) < Decision_k_DETC);
        Decision_k_DETC = double(flip(Decision_k_DETC,1) > Decision_k_DETC);

        [~,a_choose_DETC] = max(Decision_k_DETC); 
        a_choose_mask_DETC = double((Decision_k_DETC) > flip(Decision_k_DETC,1));

        reward_DETC = R(:,:,1) .* a_choose_mask_DETC;
        Regret_DETC = Regret_DETC + (r_opt - r(a_choose_DETC));
        Nk_DETC = Nk_DETC + a_choose_mask_DETC;

        t_DETC_2 = 1;
        r_bar_DETC_2  = sum(reward_DETC);
        Regret_record_DETC(t) = mean(Regret_DETC);
    else
        Stopping_DETC_2 = double(abs(r_prime_DETC - r_bar_DETC_2) > sqrt(2 / t_DETC_2 * log( T / t_DETC_2 * ( (log(T/t_DETC_2))^2+1) ))) .* (1-Stop_DETC_2);
        Stop_DETC_2 = Stop_DETC_2 + Stopping_DETC_2;
        a_choose_end_DETC = repmat((r_prime_DETC > r_bar_DETC_2),2,1) .* CurBest_k_DETC + repmat((r_prime_DETC <= r_bar_DETC_2),2,1) .* flip(CurBest_k_DETC,1);
        Decision_k_DETC = Stopping_DETC_2 .* a_choose_end_DETC + (1-Stopping_DETC_2) .* Decision_k_DETC;
        [~,a_choose_DETC] = max(Decision_k_DETC); 
        a_choose_mask_DETC = double((Decision_k_DETC) > flip(Decision_k_DETC,1));

        reward_DETC = R(:,:,1) .* a_choose_mask_DETC;
        Regret_DETC = Regret_DETC + (r_opt - r(a_choose_DETC));
        Nk_DETC = Nk_DETC + a_choose_mask_DETC;

        r_bar_DETC_2  = (sum(reward_DETC) + r_bar_DETC_2 * t_DETC_2)/(t_DETC_2 + 1);
        t_DETC_2 = t_DETC_2 + 1;
        Regret_record_DETC(t) = mean(Regret_DETC);
    end
    
    if mod(t, 10000) == 0     
        diary on
        disp(['t = ', num2str(t), ', Regret: ', num2str(mean(Regret_DETC))])
        diary off
    end
    
end

Final_Sub_Pull_DETC = Nk_DETC(2,:)';
Final_Reg_DETC = Regret_DETC;

diary on
disp(['DETC:   Regret:', num2str(mean(Final_Reg_DETC)), ',STD:', num2str(std(Final_Sub_Pull_DETC))])
diary off
end



%%
[f_1,xi_1] = ksdensity(Final_Sub_Pull,'BoundaryCorrection','log','Function','pdf','NumPoints',T/2,'Support',[0 T+1]);
[f_2,xi_2] = ksdensity(Final_Sub_Pull_UG,'BoundaryCorrection','log','Function','pdf','NumPoints',T/2,'Support',[0 T+1]);
[f_3,xi_3] = ksdensity(Final_Sub_Pull_kl,'BoundaryCorrection','log','Function','pdf','NumPoints',T/2,'Support',[0 T+1]);
[f_4,xi_4] = ksdensity(Final_Sub_Pull_klug,'BoundaryCorrection','log','Function','pdf','NumPoints',T/2,'Support',[0 T+1]);
[f_5,xi_5] = ksdensity(Final_Sub_Pull_UCB,'BoundaryCorrection','log','Function','pdf','NumPoints',T/2,'Support',[0 T+1]);
[f_6,xi_6] = ksdensity(Final_Sub_Pull_klUCB,'BoundaryCorrection','log','Function','pdf','NumPoints',T/2,'Support',[0 T+1]);
[f_7,xi_7] = ksdensity(Final_Sub_Pull_BAI,'BoundaryCorrection','log','Function','pdf','NumPoints',T/2,'Support',[0 T+1]);
[f_8,xi_8] = ksdensity(Final_Sub_Pull_DETC,'BoundaryCorrection','log','Function','pdf','NumPoints',T/2,'Support',[0 T+1]);

%%
figure;
hold on;
plot(xi_1,f_1,'LineWidth',1.5);
plot(xi_2,f_2,'LineWidth',1.5);
plot(xi_3,f_3,'LineWidth',1.5);
plot(xi_4,f_4,'LineWidth',1.5);
plot(xi_5,f_5,'LineWidth',1.5);
plot(xi_6,f_6,'LineWidth',1.5);
plot(xi_7,f_7,'LineWidth',1.5);
plot(xi_8,f_8,'LineWidth',1.5);
set(gca,'YScale','log','XLim',[25,250])
set(gca,'YLim',[1e-10,1])
legend('EOCP','EOCP-UG','KL-EOCP','KL-EOCP-UG','UCB','KL-UCB','BAI-ETC','DETC','location','best')
xlabel('Number of Pulls','FontSize',15,'FontName','Times New Roman')
ylabel('Probability Distribution','FontSize',20,'FontName','Times New Roman')

%% 
figure;
hold on;
plot([1:T], Regret_record,'LineWidth',1.8)
plot([1:T], Regret_record_UG,'LineWidth',1.8)
plot([1:T], Regret_record_kl,'LineWidth',1.8)
%plot([1:T], Regret_record_klug,'LineWidth',1.8)
plot([1:T], Regret_record_UCB,'LineWidth',1.8)
plot([1:T], Regret_record_klUCB,'LineWidth',1.8)
plot([1:T], Regret_record_BAI,'LineWidth',1.8)
plot([1:T], Regret_record_DETC,'LineWidth',1.8)
set(gca,'Xscale','log')
legend('EOCP','EOCP-UG','KL-EOCP','UCB','KL-UCB','BAI-ETC','DETC','location','best')
xlabel('Number of Rounds','FontSize',15,'FontName','Times New Roman')
ylabel('Regret','FontSize',20,'FontName','Times New Roman')
%%
% save('dataBer.mat','Final_Sub_Pull','Final_Sub_Pull_UG','Final_Sub_Pull_kl','Final_Sub_Pull_klug',...
%     'Final_Sub_Pull_UCB','Final_Sub_Pull_klUCB','Final_Sub_Pull_BAI','Final_Sub_Pull_DETC',...
%     'Regret_record','Regret_record_UG', 'Regret_record_kl','Regret_record_klug',...
%     'Regret_record_UCB','Regret_record_klUCB','Regret_record_BAI','Regret_record_DETC')

%%
figure;
hold on;
plot([1:T], Regret_record_UCB,'LineWidth',1.8)
plot([1:T], Regret_record_klUCB,'LineWidth',1.8)

plot([1:T], Regret_record,'LineWidth',1.8)
plot([1:T], Regret_record_kl,'LineWidth',1.8)
set(gca,'Xscale','log')
legend('Vanilla UCB','KL-UCB','EOCP','KL-EOCP','location','best')
xlabel('Number of Rounds','FontSize',15,'FontName','Times New Roman')
ylabel('Regret','FontSize',20,'FontName','Times New Roman')