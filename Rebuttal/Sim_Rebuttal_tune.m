clear all; close all; 

rng(1)

A = 2;
T = 100000 ;                   % total number
M = 1 ;                    % batch size
N = 2;                     % offline number
mu = [0.5, 0.5];            % data coverage
w = 1;                    % weight
alpha = 1;

r = [0.7, 0.2];                       % reward
r_opt = max(r);                     
iternum = 100000;
logbns = log(T) + sqrt(log(T));
Monitor = 10000;


R_off = zeros(2,iternum,N);                 % offline data
R_off(1,:,:) = randn(iternum,N) + r(1);
R_off(2,:,:) = randn(iternum,N) + r(2);

%% eocp Parameter
diary 'mylog3.txt'
rng(1)

Nk = N * mu' .* ones(2,iternum);                          % number of pulls
    
r_bar = mean(R_off,3);   % empirical mean
bk = sqrt(2 * alpha * logbns ./ Nk);                             % bonus
Decision_k = r_bar + bk;
Regret_record = zeros(1,T);
Regret = zeros(1,iternum);

for t=1:T
    
    R = zeros(2, iternum, 1);                   % online sample
    R(1,:,:) = randn(iternum,1) + r(1);
    R(2,:,:) = randn(iternum,1) + r(2);


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
    if mod(t, Monitor) == 0     
        diary on
        disp(['t = ', num2str(t), ', Regret: ', num2str(mean(Regret))])
        diary off
    end
    
end

%%
Final_Sub_Pull = Nk(2,:)';
Final_Reg = Regret;

diary on
disp(['Prop:   Regret:', num2str(mean(Final_Reg)), ',STD:', num2str(std(Final_Sub_Pull))])
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
    R(1,:,:) = randn(iternum,1) + r(1);
    R(2,:,:) = randn(iternum,1) + r(2);

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
    if mod(t, Monitor) == 0     
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
    R(1,:,:) = randn(iternum,1) + r(1);
    R(2,:,:) = randn(iternum,1) + r(2);

    Decision_k_UCB = r_bar_UCB + bk_UCB;                            % bonus
                                 
    [~,a_choose_UCB] = max(Decision_k_UCB); 
    a_choose_mask_UCB = double((Decision_k_UCB) > flip(Decision_k_UCB,1));
    reward_UCB = R(:,:,1) .* a_choose_mask_UCB;

    Regret_UCB = Regret_UCB + (r_opt - r(a_choose_UCB));

    r_bar_UCB = (r_bar_UCB .* Nk_UCB + reward_UCB) ./ (Nk_UCB + a_choose_mask_UCB);          % empirical mean
    Nk_UCB = Nk_UCB + a_choose_mask_UCB;
    bk_UCB = sqrt(2 * alpha * log(T) ./ Nk_UCB); 
    Regret_record_UCB(t) = mean(Regret_UCB);

    if mod(t, Monitor) == 0     
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
    R(1,:,:) = randn(iternum,1) + r(1);
    R(2,:,:) = randn(iternum,1) + r(2);


    Stopping_BAI = repmat( double( abs(r_bar_BAI(1,:) - r_bar_BAI(2,:)) > sqrt(8.2*log(T/t)/t) ), 2,1) .* (1 - Stop_BAI);
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
    if mod(t, Monitor) == 0     
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


%% DETC_tune1

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
T_one = 1 * log(T)^2;

for t=1:T
    
    R = zeros(2, iternum, 1);                   % online sample
    R(1,:,:) = randn(iternum,1) + r(1);
    R(2,:,:) = randn(iternum,1) + r(2);

    if t < ceil(T_one) 
        Stopping_DETC = repmat( double( abs(r_bar_DETC(1,:) - r_bar_DETC(2,:)) > sqrt(16 * max(log(T_one / t),0) / t) ), 2,1) .* (1 - Stop_DETC);
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

    elseif t == ceil(T_one) 
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
        Stopping_DETC_2 = double(abs(r_prime_DETC - r_bar_DETC_2) > sqrt(2 / t_DETC_2 * log( T / t_DETC_2 * ( (log(T/t_DETC_2))^2+1 ) ))) .* (1-Stop_DETC_2);
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
    
    if mod(t, Monitor) == 0     
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




%% DETC_tune_2

if 1

rng(1)

Nk_DETC = N * mu' .* ones(2,iternum);                          % number of pulls
r_bar_DETC = mean(R_off,3);   % empirical mean
bk_DETC = sqrt(2 * alpha * log(T) ./ Nk_DETC);                             % bonus

t_DETC_2 = 1;
r_bar_DETC_2 = zeros(1,iternum);
r_prime_DETC = zeros(1,iternum);

Decision_k_DETC = r_bar_DETC + bk_DETC;
Regret_record_DETC_2 = zeros(1,T);
Regret_DETC = zeros(1,iternum);
Stop_DETC = zeros(2,iternum);
Stop_DETC_2 = zeros(1,iternum);
T_one = 1.6 * log(T)^2;

for t=1:T
    
    R = zeros(2, iternum, 1);                   % online sample
    R(1,:,:) = randn(iternum,1) + r(1);
    R(2,:,:) = randn(iternum,1) + r(2);

    if t < ceil(T_one) 
        Stopping_DETC = repmat( double( abs(r_bar_DETC(1,:) - r_bar_DETC(2,:)) > sqrt(16 * max(log(T_one / t),0) / t) ), 2,1) .* (1 - Stop_DETC);
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
        Regret_record_DETC_2(t) = mean(Regret_DETC);

    elseif t == ceil(T_one) 
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
        Regret_record_DETC_2(t) = mean(Regret_DETC);
    else
        Stopping_DETC_2 = double(abs(r_prime_DETC - r_bar_DETC_2) > sqrt(1 / t_DETC_2 * log( T / t_DETC_2 * ( (log(T/t_DETC_2))^2+1 ) ))) .* (1-Stop_DETC_2);
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
        Regret_record_DETC_2(t) = mean(Regret_DETC);
    end
    
    if mod(t, Monitor) == 0     
        diary on
        disp(['t = ', num2str(t), ', Regret: ', num2str(mean(Regret_DETC))])
        diary off
    end
    
end

Final_Sub_Pull_DETC_2 = Nk_DETC(2,:)';
Final_Reg_DETC_2 = Regret_DETC;

diary on
disp(['DETC_2:   Regret:', num2str(mean(Final_Reg_DETC_2)), ',STD:', num2str(std(Final_Sub_Pull_DETC_2))])
diary off
end


%%
% [f_1,xi_1] = ksdensity(Final_Sub_Pull,'BoundaryCorrection','log','Function','pdf','NumPoints',T/2,'Support',[0 T+1]);
% [f_2,xi_2] = ksdensity(Final_Sub_Pull_UG,'BoundaryCorrection','log','Function','pdf','NumPoints',T/2,'Support',[0 T+1]);
% [f_3,xi_3] = ksdensity(Final_Sub_Pull_UCB,'BoundaryCorrection','log','Function','pdf','NumPoints',T/2,'Support',[0 T+1]);
% [f_4,xi_4] = ksdensity(Final_Sub_Pull_BAI,'BoundaryCorrection','log','Function','pdf','NumPoints',T/2,'Support',[0 T+1]);
% [f_5,xi_5] = ksdensity(Final_Sub_Pull_DETC,'BoundaryCorrection','log','Function','pdf','NumPoints',T/2,'Support',[0 T+1]);
% 
% %%
% figure;
% hold on;
% plot(xi_1,f_1);
% plot(xi_2,f_2);
% plot(xi_3,f_3);
% plot(xi_4,f_4);
% plot(xi_5,f_5);
% set(gca,'YScale','log','XLim',[0,1000])
% set(gca,'YLim',[1e-10,1])
% legend('EOCP','EOCP-UG','UCB','BAI-ETC','DETC','location','best')

%% 
figure;
hold on;
plot([1:T], Regret_record,'LineWidth',1.8)
plot([1:T], Regret_record_UG,'LineWidth',1.8)
plot([1:T], Regret_record_UCB,'LineWidth',1.8)
plot([1:T], Regret_record_BAI,'LineWidth',1.8)
plot([1:T], Regret_record_DETC,'LineWidth',1.8)
plot([1:T], Regret_record_DETC_2,'LineWidth',1.8)
set(gca,'Xscale','log')
%set(gca,'Ylim',[0,200])
legend('EOCP','EOCP-UG','UCB','BAI-ETC','DETC-1Param','DETC-2Param','location','best')
xlabel('Number of Rounds','FontSize',15,'FontName','Times New Roman')
ylabel('Regret','FontSize',20,'FontName','Times New Roman')

%%
save('data_tune.mat','Final_Sub_Pull','Final_Sub_Pull_UG','Final_Sub_Pull_UCB','Final_Sub_Pull_BAI','Final_Sub_Pull_DETC','Final_Sub_Pull_DETC_2'...
    ,'Regret_record','Regret_record_UG','Regret_record_UCB','Regret_record_BAI','Regret_record_DETC','Regret_record_DETC_2')
