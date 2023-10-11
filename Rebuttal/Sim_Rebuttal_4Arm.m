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
iternum = 10000;
logbns = log(T) + 0.2*sqrt(log(T));
Monitor = 10000;


R_off = zeros(A,iternum,N);                 % offline data
R_off(1,:,:) = randn(iternum,N) + r(1);
R_off(2,:,:) = randn(iternum,N) + r(2);
R_off(3,:,:) = randn(iternum,N) + r(3);
R_off(4,:,:) = randn(iternum,N) + r(4);

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
    R(1,:,:) = randn(iternum,1) + r(1);
    R(2,:,:) = randn(iternum,1) + r(2);
    R(3,:,:) = randn(iternum,1) + r(3);
    R(4,:,:) = randn(iternum,1) + r(4);


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
    R(1,:,:) = randn(iternum,1) + r(1);
    R(2,:,:) = randn(iternum,1) + r(2);
    R(3,:,:) = randn(iternum,1) + r(3);
    R(4,:,:) = randn(iternum,1) + r(4);
    
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
    R(1,:,:) = randn(iternum,1) + r(1);
    R(2,:,:) = randn(iternum,1) + r(2);
    R(3,:,:) = randn(iternum,1) + r(3);
    R(4,:,:) = randn(iternum,1) + r(4);

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

%% BAI

if 1

rng(1)

Nk_BAI = N * mu' .* ones(A,iternum);                          % number of pulls
    
r_bar_BAI = mean(R_off,3);   % empirical mean
bk_BAI = sqrt(2 * alpha * log(T) ./ Nk_BAI);                             % bonus
Decision_k_BAI = r_bar_BAI + bk_BAI;
Regret_record_BAI = zeros(1,T);
Regret_BAI = zeros(1,iternum);
Stop_BAI = zeros(A,iternum);

for t=1:T
    
    R = zeros(A, iternum, 1);                   % online sample
    R(1,:,:) = randn(iternum,1) + r(1);
    R(2,:,:) = randn(iternum,1) + r(2);
    R(3,:,:) = randn(iternum,1) + r(3);
    R(4,:,:) = randn(iternum,1) + r(4);
    
    
    Z1_all = (repmat(r_bar_BAI(1,:),3,1) - r_bar_BAI(2:4,:)).^2 .* (repmat(Nk_BAI(1,:),3,1) + Nk_BAI(2:4,:));
    Z2_all = (repmat(r_bar_BAI(2,:),3,1) - r_bar_BAI([1,3,4],:)).^2 .* (repmat(Nk_BAI(2,:),3,1) + Nk_BAI([1,3,4],:));
    Z3_all = (repmat(r_bar_BAI(3,:),3,1) - r_bar_BAI([1,2,4],:)).^2 .* (repmat(Nk_BAI(3,:),3,1) + Nk_BAI([1,2,4],:));
    Z4_all = (repmat(r_bar_BAI(4,:),3,1) - r_bar_BAI([1,2,3],:)).^2 .* (repmat(Nk_BAI(4,:),3,1) + Nk_BAI([1,2,3],:));
    Z(1,:) = min(Z1_all);
    Z(2,:) = min(Z2_all);
    Z(3,:) = min(Z3_all);
    Z(4,:) = min(Z4_all);
    Z_end = max(Z);
    
    
    Stopping_BAI = repmat( double( Z_end > 12*log(T/t) ), A,1) .* (1 - Stop_BAI);
    Stop_BAI = Stop_BAI + Stopping_BAI;
    
    Prob = [sqrt(3), 1 ,1 ,1] / (3+sqrt(3));
    ArmPull_k_BAI = randsrc(1,iternum,[1,2,3,4; Prob ]);
    IndexPull_k_BAI = [0:A:A*(iternum-1)] + ArmPull_k_BAI;
    
    ArmPull_k_BAI_mask = zeros(A,iternum);
    ArmPull_k_BAI_mask(IndexPull_k_BAI) = 1;
    
    Decision_k_BAI = Stop_BAI .* Decision_k_BAI + (1 - Stop_BAI).* ArmPull_k_BAI_mask;
    Decision_k_BAI = Stopping_BAI .* r_bar_BAI + (1 - Stopping_BAI).* Decision_k_BAI;
    

    [~,a_choose_BAI] = max(Decision_k_BAI); 
    a_choose_mask_BAI = (Decision_k_BAI ==(max(Decision_k_BAI)));
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
        R = randn(4,1) + r';
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

%% 
figure;
hold on;
plot([1:T], Regret_record,'LineWidth',1.8)
plot([1:T], Regret_record_UG,'LineWidth',1.8)
plot([1:T], Regret_record_UCB,'LineWidth',1.8)
plot([1:T], Regret_record_BAI,'LineWidth',1.8)
plot([1:T], Regret_record_AE,'LineWidth',1.8)
%plot([1:T], Regret_record_DETC,'LineWidth',1.8)
%plot([1:T], Regret_record_DETC_2,'LineWidth',1.8)
set(gca,'Xscale','log')
%set(gca,'Ylim',[0,250])
legend('EOCP','EOCP-UG','UCB','BAI-ETC','Action-Elimination','location','best')
xlabel('Number of Rounds','FontSize',15,'FontName','Times New Roman')
ylabel('Regret','FontSize',20,'FontName','Times New Roman')
%%
save('data_multi.mat','Regret_record','Regret_record_UG','Regret_record_UCB','Regret_record_BAI','Regret_record_AE')

