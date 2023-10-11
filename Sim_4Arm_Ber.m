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
R_off(1,:,:) = double(rand(iternum,N) < r(1));
R_off(2,:,:) = double(rand(iternum,N) < r(2));
R_off(3,:,:) = double(rand(iternum,N) < r(3));
R_off(4,:,:) = double(rand(iternum,N) < r(4));

%% eocp Parameter
diary 'mylog3.txt'
rng(1)

Nk = N * mu' .* ones(A,iternum);                          % number of pulls
    
r_bar = mean(R_off,3);   % empirical mean
bk = sqrt(2 * alpha * logbns ./ Nk);                             % bonus
Decision_k = r_bar + bk;
Regret_record = zeros(1,T);
Regret = zeros(1,iternum);

for t=1:T
    
    R = zeros(A, iternum, 1);                   % online sample
    R(1,:,:) = double(rand(iternum,1) < r(1));
    R(2,:,:) = double(rand(iternum,1) < r(2));
    R(3,:,:) = double(rand(iternum,1) < r(3));
    R(4,:,:) = double(rand(iternum,1) < r(4));


    if t < 8*A * alpha  * logbns / Deltamin^2
        Decision_k = r_bar + bk;                            % bonus
                                 
    elseif t-1 < 8*A * alpha  * logbns / Deltamin^2
        Decision_k = r_bar - bk;
    end

    [~,a_choose] = max(Decision_k); 
    a_choose_mask = Decision_k==(max(Decision_k));
    
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

Final_Reg = Regret;

diary on
disp(['Prop:   Regret:', num2str(mean(Final_Reg)), ',STD:', num2str(std(Final_Reg))])
diary off

%% ug Parameter
rng(1)

Nk_UG = N * mu' .* ones(A,iternum);                          % number of pulls
    
r_bar_UG = mean(R_off,3);   % empirical mean
bk_UG = sqrt(2 * alpha * logbns ./ Nk_UG);                             % bonus
Decision_k_UG = r_bar_UG + bk_UG;
Regret_record_UG = zeros(1,T);
Regret_UG = zeros(1,iternum);
Stop_UG = zeros(A,iternum);

for t=1:T
    
    R = zeros(A, iternum, 1);                   % online sample
    R(1,:,:) = double(rand(iternum,1) < r(1));
    R(2,:,:) = double(rand(iternum,1) < r(2));
    R(3,:,:) = double(rand(iternum,1) < r(3));
    R(4,:,:) = double(rand(iternum,1) < r(4));
    
    Nk_sort_UG = sort(Nk_UG,'descend');
    MostPull_UG = Nk_sort_UG(1,:);
    SecondMostPull_UG = Nk_sort_UG(2,:);

    Stopping_UG = repmat( double(MostPull_UG ./ SecondMostPull_UG > logbns), A,1) .* (1 - Stop_UG);
    Stop_UG = Stop_UG + Stopping_UG;
    Decision_k_UG = Stop_UG .* Decision_k_UG + (1 - Stop_UG).* (r_bar_UG + bk_UG);
    Decision_k_UG = Stopping_UG .* (r_bar_UG - bk_UG) + (1 - Stopping_UG).* Decision_k_UG;

    [~,a_choose_UG] = max(Decision_k_UG); 
    a_choose_mask_UG = Decision_k_UG==(max(Decision_k_UG));
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

Nk_UCB = N * mu' .* ones(A,iternum);                          % number of pulls
    
r_bar_UCB = mean(R_off,3);   % empirical mean
bk_UCB = sqrt(2 * alpha * log(T) ./ Nk_UCB);                             % bonus
Decision_k_UCB = r_bar_UCB + bk_UCB;
Regret_UCB = zeros(1,iternum);
Regret_record_UCB = zeros(1,T);

for t=1:T

    R = zeros(A, iternum, 1);                   % online sample
    R(1,:,:) = double(rand(iternum,1) < r(1));
    R(2,:,:) = double(rand(iternum,1) < r(2));
    R(3,:,:) = double(rand(iternum,1) < r(3));
    R(4,:,:) = double(rand(iternum,1) < r(4));

    Decision_k_UCB = r_bar_UCB + bk_UCB;                            % bonus
                                 
    [~,a_choose_UCB] = max(Decision_k_UCB); 
    a_choose_mask_UCB = Decision_k_UCB==(max(Decision_k_UCB));
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

Final_Reg_UCB = Regret_UCB;

diary on
disp(['UCB:   Regret:', num2str(mean(Final_Reg_UCB)), ',STD:', num2str(std(Final_Reg_UCB))])
diary off
end


%% AE
iternum_AE = 1000;
Regret_record_AE_all = zeros(iternum_AE,T);
for iter = 1:iternum_AE
    Empirical_Offline = mean(R_off,3);
    r_bar_AE = Empirical_Offline(:,iter);
    Nk_AE = N * mu';
    bk_AE = sqrt(2 * alpha * log(T) ./ Nk_AE);  
    Regret_AE = 0;
    Eliminate_Action = [0;0;0;0];
    for t=1:T
        R = double(rand(4,1) < r');
        ActionSetSize = A - sum(Eliminate_Action);
        ActionRaw = mod(t, ActionSetSize) + 1;
        flag =0;
        for a = 1:A
            if Eliminate_Action(a) == 0
                flag = flag + 1;
            end
            if flag == ActionRaw
                Action = a;
                break;
            end
        end    

        Reward_AE = R(Action);
        Regret_AE = Regret_AE + (r_opt - r(Action));
        Regret_record_AE_all(iter,t) = Regret_AE;

        r_bar_AE(Action) = (r_bar_AE(Action) * Nk_AE(Action) + Reward_AE) / (Nk_AE(Action) + 1);
        Nk_AE(Action) = Nk_AE(Action) + 1;
        bk_AE = sqrt(2 * alpha * log(T) ./ Nk_AE);  

        [r_CurHigh, CurBest] = max(r_bar_AE);
        for a = 1:A
            if Eliminate_Action(a) == 0 && a~= CurBest 
                if r_bar_AE(CurBest)- bk_AE(CurBest) > r_bar_AE(a)+ bk_AE(a)
                    Eliminate_Action(a) = 1;
                end
            end
        end

    end
    
end
Regret_record_AE = mean(Regret_record_AE_all);
Final_Reg_AE = Regret_record_AE_all(:,end);
disp(['AE:   Regret:', num2str(mean(Final_Reg_AE)), ',STD:', num2str(std(Final_Reg_AE))])
%% KL-UCB
if 1

rng(1)

Nk_klUCB = N * mu' .* ones(A,iternum);                          % number of pulls
    
r_bar_klUCB = mean(R_off,3);   % empirical mean
[UCB_klUCB, LCB_klUCB] = KL_ULCB(r_bar_klUCB, Nk_klUCB, log(T));

Decision_k_klUCB = UCB_klUCB;
Regret_klUCB = zeros(1,iternum);
Regret_record_klUCB = zeros(1,T);

for t=1:T

    R = zeros(A, iternum, 1);                   % online sample
    R(1,:,:) = double(rand(iternum,1) < r(1));
    R(2,:,:) = double(rand(iternum,1) < r(2));
    R(3,:,:) = double(rand(iternum,1) < r(3));
    R(4,:,:) = double(rand(iternum,1) < r(4));

    Decision_k_klUCB = UCB_klUCB;                            % bonus
                                 
    [~,a_choose_klUCB] = max(Decision_k_klUCB); 
    a_choose_mask_klUCB = Decision_k_klUCB==(max(Decision_k_klUCB));
    reward_klUCB = R(:,:,1) .* a_choose_mask_klUCB;

    Regret_klUCB = Regret_klUCB + (r_opt - r(a_choose_klUCB));

    r_bar_klUCB = (r_bar_klUCB .* Nk_klUCB + reward_klUCB) ./ (Nk_klUCB + a_choose_mask_klUCB);          % empirical mean
    Nk_klUCB = Nk_klUCB + a_choose_mask_klUCB;
    [UCB_klUCB, LCB_klUCB] = KL_ULCB(r_bar_klUCB, Nk_klUCB, log(T));
    Regret_record_klUCB(t) = mean(Regret_klUCB);

    if mod(t, Monitor) == 0     
        diary on
        disp(['t = ', num2str(t), ', Regret: ', num2str(mean(Regret_klUCB))])
        diary off
    end

end

Final_Reg_klUCB = Regret_klUCB;

diary on
disp(['kl-UCB:   Regret:', num2str(mean(Final_Reg_klUCB)), ',STD:', num2str(std(Final_Reg_klUCB))])
diary off
end

%% KL-EOCP
rng(1)

KL_mean1 = r(2) .* log(r(2)/r(1)) + (1-r(2)).* log( (1-r(2))./(1-r(1)) ) ;
[~,ra_prime] = invKL_ULCB(r(1),1,KL_mean1/4); 
KL_mean2 = 4 * (ra_prime .* log(ra_prime/r(2) ) + (1-ra_prime) .* log( (1-ra_prime)./(1-r(2)) ) ) ;
KL_mean = min(KL_mean1, KL_mean2);

Nk_kl = N * mu' .* ones(A,iternum);                          % number of pulls
    
r_bar_kl = mean(R_off,3);   % empirical mean
[UCB_kl, LCB_kl] = KL_ULCB(r_bar_kl, Nk_kl, logbns);

Decision_k_kl = UCB_kl;
Regret_record_kl = zeros(1,T);
Regret_kl = zeros(1,iternum);

for t=1:T
    
    R = zeros(A, iternum, 1);                   % online sample
    R(1,:,:) = double(rand(iternum,1) < r(1));
    R(2,:,:) = double(rand(iternum,1) < r(2));
    R(3,:,:) = double(rand(iternum,1) < r(3));
    R(4,:,:) = double(rand(iternum,1) < r(4));


    if t < 8/KL_mean * alpha  * logbns
        Decision_k_kl = UCB_kl;                            % bonus
                                 
    elseif t-1 < 8/KL_mean * alpha  * logbns
        Decision_k_kl = LCB_kl;
    end

    [~,a_choose_kl] = max(Decision_k_kl); 
    a_choose_mask_kl = Decision_k_kl==(max(Decision_k_kl));
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

%% 
load("DETCK_Ber.mat")
load('TAS_ETC.mat')
figure;
hold on;
plot([1:T], Regret_record,'LineWidth',1.8)
plot([1:T], Regret_record_UG,'LineWidth',1.8)
plot([1:T], Regret_record_UCB,'LineWidth',1.8)
plot([1:T], Regret_record_BAI,'LineWidth',1.8)
plot([1:T], Regret_record_AE,'LineWidth',1.8)
plot([1:T], Regret_mean_DETC,'LineWidth',1.8)
plot([1:T], Regret_record_klUCB,'LineWidth',1.8)
plot([1:T], Regret_record_kl,'LineWidth',1.8)
set(gca,'Xscale','log')
set(gca,'Ylim',[0,220])
legend('EOCP','EOCP-UG','UCB','TAS-ETC','Action-Elimination','DETC-K','KL-UCB','KL-EOCP','location','best')
xlabel('Number of Rounds','FontSize',15,'FontName','Times New Roman')
ylabel('Regret','FontSize',20,'FontName','Times New Roman')
%%
% save('data_Ber.mat','Regret_record','Regret_record_UG','Regret_record_UCB','Regret_record_AE','Regret_record_klUCB','Regret_record_kl')

